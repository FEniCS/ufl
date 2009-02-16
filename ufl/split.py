"Algorithm for splitting a Function into sub functions."

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2009-02-16"

# Modified by Anders Logg, 2008

from ufl.log import error
from ufl.assertions import ufl_assert
from ufl.common import product
from ufl.finiteelement import MixedElement
from ufl.tensors import as_vector, as_matrix, as_tensor

def split(v):
    "Split Function into its sub Functions if any."
    
    # Special case: simple element, return function
    element = v.element()
    if not isinstance(element, MixedElement):
        return (v,)
    
    # Compute value size
    value_size = product(element.value_shape())
    
    # Extract sub functions
    offset = 0
    sub_functions = []
    for i, e in enumerate(element.sub_elements()):
        shape = e.value_shape()
        rank = len(shape)
        
        if rank == 0:
            subv = v[offset]
            offset += 1
        
        elif rank == 1:
            sub_size, = shape
            
            components = []
            for j in range(sub_size):
                jj = offset + j
                components.append(v[jj])
            #components = [v[j] for j in range(offset, offset + sub_size)]
            
            subv = as_vector(components)
            offset += sub_size
        
        elif rank == 2:
            # FIXME: Handle symmetries here?
            sub_size = product(shape)
            
            components = []
            for i in range(shape[0]):
                vs = []
                for j in range(shape[1]):
                    k = offset + i*shape[1] + j
                    vs.append(v[k])
                components.append(vs)
            
            subv = as_matrix(components)
            offset += sub_size
        
        else:
            # FIXME: Handle rank > 2.
            error("Don't know how to split functions with sub functions of rank %d (yet)." % rank)
            #for indices in compute_indices(shape):
            #    #k = offset + sum(i*s for (i,s) in zip(indices, shape[1:] + (1,)))
            #    vs.append(v[indices])
        
        sub_functions.append(subv)
    
    ufl_assert(value_size == offset, "Logic breach in function splitting.") # FIXME: Not true with symmetries.
    
    return tuple(sub_functions)

