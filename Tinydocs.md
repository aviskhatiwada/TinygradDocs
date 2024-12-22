tensor.py

This file initalizes the base class "Function" defining attributes necessary for "passes": self.(metadata |  needs\_input\_grad)
	with an apply classmethod, reponsible for creating new Tensor objects which the original values undergoing application by 
	the named function. Context (ctx) under the method is defined as the device belonging to the object, its data and metadata



Dataclasses/DTypes are defined in relation to 
	(priority, itemsize, name, [optional]format, count)
	: they are represented, in the scheduler/elsewhere with the following attirbution, depending on
	: whether there is more than one of the given type.
	preface if count>1 + name if count=1 , otherwise the SCALAR\_NAME + str(count) if count >1

