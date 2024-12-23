**Synopsis of "tinygrad-notes" by mesozoic-egg:**

**1.**
Compilation is performed by creation of an Abstract-Syntax-Tree, at the top of which is a universal Memory-Buffer branch. For the given example (the multiplication of two tensors against the 0th dimension), LOAD, MUL, SUM, and STORE are the Ops that populate the AST. These Ops are afterwards linearized in order to be translated by the codegen utility. On being linearized, these Ops represent variable definitions (DEFINE\_GLOBAL), accumulator definitions (DEFINE\_ACC), integer/constant definitions, arithmetic operations (ALU), loop definitions (LOOP, ENDLOOP), and stores (STORE).  

**2.**
LazyData can be either concrete (defined with one or several arithmetic + logical Ops) or non-concerete (which warp around a concrete op; the permute Op is given as an example). The non-concerete LazyData has a \_base attribute that points to the unevaluated LazyData, the \_base of the concerete LazyData points to itself. Thus, for the example provided in (1)

Memory must be loaded into the GPU so that the pointers of the compiled C code can access the correct symbols. ScheduleItems are used to perform read, write, and run operations on the 
kernels.

The Schedule (list) is created after 


**Descriptions By Files:**
tensor.py

This file initalizes the base class "Function" defining attributes necessary for "passes": self.(metadata |  needs\_input\_grad).
	The "apply" classmethod is reponsible for creating new Tensor objects which the original values undergoing application by 
	the named function. Context (ctx) under the method is defined as the device belonging to the object, and the objects data/metadata.

The Tensor class initalizes similar parameters to the Function class; initalization begins with several instance checks; subclasses include "train" and "test", and
	its properties include .device, .shape, and .dtype. Scheduling is handled by methods .schedule\_with\_vars , .schedule, and .realize
