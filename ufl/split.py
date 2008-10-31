from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-10-31"

# Modified by Anders Logg, 2008

from .finiteelement import MixedElement
from .tensors import as_vector, as_matrix, as_tensor
from .common import product
from .output import ufl_assert, ufl_error

def split(v):
    "Split function into its sub functions if any"

    # Special case: simple element, return function
    element = v.element()
    if not isinstance(element, MixedElement):
        return v
    
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
            size = product(shape)
            components = [v[j] for j in range(offset, offset + size)]
            subv = as_vector(components)
            offset += size
        else:
            # FIXME: Handle general case
            # FIXME: Handle symmetries
            size = product(shape) # FIXME: Ignoring symmetries here, is this ok?
            ufl_error("Don't know how to split functions with sub functions of rank %d (yet)." % rank)
            
            def tensor_components(sh, off):
                comp = []
                #def tensor_components(sh, off):
                #for 
                
            components = []
            for j, s in shape:
                pass #v[j] for j in range(offset, offset + size)
            subv = as_tensor(components)
            offset += size
        sub_functions.append(subv)
    
    ufl_assert(value_size == offset, "Logic breach in function splitting.")

    return tuple(sub_functions)
