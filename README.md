*** A personal diary for explaining to myself tinygrad concepts. Touring @mesozoicegg 's tutorials and the tinygrad source. ***  
**1.**
Compilation is performed by creation of an Abstract-Syntax-Tree, at the top of which is a universal Memory-Buffer branch. For the given example (the multiplication of two tensors against the 0th dimension), LOAD, MUL, SUM, and STORE are the Ops that populate the AST. These Ops are afterwards linearized in order to be translated by the codegen utility. On being linearized, these Ops represent variable definitions (DEFINE\_GLOBAL), accumulator definitions (DEFINE\_ACC), integer/constant definitions, arithmetic operations (ALU), loop definitions (LOOP, ENDLOOP), and stores (STORE).  

**2.**
LazyData can be either concrete (defined with one or several arithmetic + logical Ops) or non-concerete (which warp around a concrete op; the permute Op is given as an example). The non-concerete LazyData has a \_base attribute that points to the unevaluated LazyData, the \_base of the concerete LazyData points to itself. Thus, for the example provided in (1)
	
Memory must be loaded into the GPU so that the pointers of the compiled C code can access the correct symbols. ScheduleItems are used to perform read, write, and run operations on the 
kernels. The Schedule (list) is created after the listed lazy data is recursively parsed to find which branches have the .realize attribute set. The code below is annotated to show how 
this is carried through procedurally. In essence, the conditional recursion depends on the type of Op, among other properties, that 'buf' uses: after the inital dtype check, a check
is done on whether the buf is concrete; if not, then a check is done to determine if it is a "simple\_pad" (if the shape of the non-concrete buffer is larger than then its src, has a single
view with a mask, and the base shape is larger than the mask), or if it needs to be immediately realized (since it is not a pad); the recursion begins again with the base. Thus, this check handles ImageDtypes, Concrete-LazyData, and Non-concrete Lazydata.


```
def _recurse_lb(buf:LazyBuffer, realizes:Set[LazyBuffer], allbufs:Dict[LazyBuffer, None],
                simple_pads:Set[LazyBuffer], children:DefaultDict[LazyBuffer, Dict[LazyBuffer, None]], scheduled=False):

  # if realized, or processed, return immediately
  if buf in allbufs or buf.base.realized: return

  if GRAPH: log_lazybuffer(buf, scheduled)
  #* if the dimensons cannot be equated or improper shape, force dtypes to f32
  if isinstance(buf.dtype, ImageDType) and (prod(buf.shape) != prod(buf.dtype.shape) or
                                            not any(buf.shape[x]%4 == 0 for x in buf.st.unit_stride_axes())):
    if DEBUG >= 3: print(f"forcing image {buf.dtype} with shape {buf.shape} to float32")
    buf.dtype = dtypes.float32  # NOTE: this is what makes the dtype above not match
  #if the buffer is non-concrete/wrapper,
  if buf.base != buf: # realize all places where the buffer is expanded
    if prod(buf.base.st.shape) < prod(buf.st.shape): # if the non-concrete is larger than the base buffer,
      if len(buf.st.views) == 1 and buf.st.views[-1].mask and all_int(buf.base.st.shape) and \ 
		  # and it conforms to simple properties (single view, mask, and integer dimensions within the base),
          prod(buf.base.st.shape) >= prod([y-x for x,y in buf.st.views[-1].mask]): # with the base shape being larger than the mask of the wrapper,
        simple_pads.add(buf.base) #it is added to simple_pads, and does not need immediate realization.
      else:
        realizes.add(buf.base)
    return _recurse_lb(buf.base, realizes, allbufs, simple_pads, children) # run recursive function beginning with the base of the non-concrete lazydata.

  #other conditions for buf-realize, this is if the buffer is concrete, and not a wrapper:
  if buf.forced_realize: realizes.add(buf)
  allbufs[buf] = None
  if buf.op in LoadOps: realizes.add(buf.base)
  if buf.op == LoadOps.COPY:
    assert buf.srcs[0].st.contiguous and buf.srcs[0].size == buf.srcs[0].base.size, "can only copy contig"
    realizes.add(buf.srcs[0].base)
  for x in buf.srcs: # if source i.e., arith. operation
    children[x.base][buf] = None # src[...] w/ associated operation that has as src.
    _recurse_lb(x, realizes, allbufs, simple_pads, children) 
```

(Note: this method of recursively realizing concrete/non-concrete LBs has been deprecated.)  

The \_schedule\_one function is responsible for generating the ScheduleItems; the kind of ScheduleItem is conditional on the Op; if is a LazyOp (WAIT, SYNC, CUSTOM, EMPTY, COPY), then a LazyOp is assigned;
otherwise, its shaped is reduced if applicable (with 'reduce\_for\_op[out]'), after a recursive check, a STORE Op is utilized if the computation is to be kept.


**<ins>Tensors:</ins>**

The tensor.py file initalizes the base class "Function" defining attributes necessary for "passes": self.(metadata |  needs\_input\_grad).
	The "apply" classmethod is reponsible for creating new Tensor objects which the original values undergoing application by 
	the named function. Context (ctx) under the method is defined as the device belonging to the object, and the objects data/metadata.
The Tensor class initalizes similar parameters to the Function class; initalization begins with several instance checks; subclasses include "train" and "test", and
	its properties include .device, .shape, and .dtype. Scheduling is handled by methods .schedule\_with\_vars , .schedule, and .realize.

**<ins>Device</ins>**

This class is responsible for handling compilation/buffer allocation; canonicalization is when a given device is made able to refer to class objects within the appropriate file (within /runtime/ops_*.py) that pertains to the device performing data operations. For instance, when a Buffer from tinygrad.devices is being allocated, it calls `Device.__getitem__` through `[]` to get the appropriate allocator:  
```
  def allocate(self, opaque=None, external_ptr=None) -> Buffer:
    assert not self.is_allocated(), "can't allocate already allocated buffer"
    self.allocator = Device[self.device].allocator
```

Thus when the Buffer is initalized and allocated like so,   
```
buf=Buffer(device='CLANG', size=1,  dtype=dtypes.int8, options=BufferSpec(cpu_access=False)).allocate()
print(tinygrad.helpers.from_mv(buf.as_buffer())[0]); del buf # -> \x00 copyes out a byte, though a (char *) cast.
```

The Device will be referenced through __get_canonicalized_item(ix), which takes `ix` as the device that will handle device-specific importing (via importlib and inspect)
```
  def __get_canonicalized_item(self, ix:str) -> Compiled:
    cpn = multiprocessing.current_process().name
    assert (cpn == "MainProcess") or ix.split(":")[0] in ["DISK", "NPY", "PYTHON"], f"can only open device {ix} from parent, not {cpn}"
    x = ix.split(":")[0].upper()
    ret = [cls for cname, cls in inspect.getmembers(importlib.import_module(f'{__name__.split(".")[0]}.runtime.ops_{x.lower()}')) \
           if (cname.lower() == x.lower() + "device")][0](ix)
    if DEBUG >= 1: print(f"opened device {ix} from pid:{os.getpid()}")
    self._opened_devices.add(ix)
    return ret #these classes specific to device/device compilers will be returned
```

Note: `tinygrad.helpers` is a module which includes miscellaneous convience functions, such as cache/data retrieval, Tempfile IO, and metrics objects. For instance, the simplified 
helper.tqdm class records metrics on completion through iterables, and the like. Much like tqdm.tqdm, SI() and HMS() are implemented within the tqdm class. Respectively, these methods represent 
byte counts according to math.log(size, 1e3), and record time through modulos/divisons by powers of 60. A simple example is shown below.

```
iterable=[_ for _ in range(10)]
tqdm=tinygrad.helpers.tqdm(iterable=iterable, desc="list iterations", unit='  iterations', unit_scale=True)
for n in range(len(iterable)): tqdm.update(n=1,close=True)
```
Similarly, a `Profiling` class is also within `helpers`, offering a simple way to access the relative speed of specified functions, perhaps for debugging purposes. A simple use of it is shown here:

```
import os,pstat
PROFILE_FN="out.proc"
with helpers.Profiling(enabled=True, fn=(fn:=os.path.join(os.getcwd(),PROFILE_FN) if PROFILE_FN else None)): time.sleep(1)
pstats.Stats(PROFILE_FN).strip_dirs().sort_stabts("tottime").print_stats(1) # this returns the profiled function that had taken the longest time per-call (ref. "tottime")
```
In the next article, about the utility of `@TinyJit`, some of the basics about GPU compilation is touched on- specifically just how the executable code is created on a GPU device. In Metal, the MSL is 
used to abstract operations that are to be performed concurrently along different thread indexes. The MTLDevice is called, and is used to call functions from MTLLibraries (MTLLirbaries are created through
calls from the Device object; the DeviceAdder is responsible for handling the functions / operations that are done in the GPU). The Metal `Command Buffer` is responsible for doing the compilation, and is constructed
through a `Command Encoder` which takes in, as arguments, all relevant bufferes, thread/threadgroup-counts, PSOs needed for the computation.  

In the source, the `.get_runner` method is responsible for handling cache hits (through the globally defined method_cache, through `.get(ckey)` and `.get(bkey)`. JIT functions mainly to reload the command queue once a 
repeated commmand buffer is recognized; a similar kind of caching is done when identical UOps are realized. @Jit functions cannot be nested, thus the conditionality of the JIT in the `llama` example, where an argument
specifies whether the `.forward` pass will be decorated by TinyJit; this will prevent a nesting error if the `.train` method has already been decorated.  
Because of Serial JIT execution, inerleaven sharding/"batching" may be problmeatic due to a device-check in JIT. Even if the same operation is done in different devices (i.e., METAL and METAL:1), the command graph  
may not be reused due to the device switch affecting `can_share_graph`, or the batched kernel being recognized as only affecting a single device.


**<ins>Loop Unrolling</ins>**  
The kernel operations used in adjusting the locality of loop iterations, and the issuing of global (gid) and local (lid) thread blocks and groups include codegen.kernel.ops.UNROLL and codegen.kernel.ops.UPCAST.  
The UPCAST Op. determines how many operations are done by a single kernel, while UNROLL fixes the number of iterations, when an operation is performed across a given axis (ie., .UNROLL(axis, amt)).
Hence, in the given example, upcasting with arguments (0,8) on a summation operations of two (4,4) tensors adjusts gix only to (0-1), with the gidx0 pointer being gix << 3. The number of threads can also
be specified by OptOps.LOCAL, which determines the amount of values that are to be locally retrieved by a given number of threads. OptOps.GROUP and OptOps.GROUP are able to use shared memory among threads;
in given first example (with .GROUP), the tensor is accessed through strides of 4, and the reduce moves by 2, and two threads access one value each; these are summed and put

```
(all even indexes are handled by t0, odd ones t1)
Tensor.ones(4,4) ->  (gid may represent "rows" while rid represents row-halves; each part of the half is carried by threads l0 and l1)
  for loop ( 
    * at gid=0, lid=0: 
      * at rid=0, acc=1 [Tensor[0][0]] --> thread0 
    * at gid=0, lid=0:
      *  at rid=1, acc=2 [Tensor[0][2] --> thread0 
      } :: the other elements of the row are carried by l1;  
```
After the `threadgroup_barrier` only l0 is used to compute the sum of the values for each reduced row.
