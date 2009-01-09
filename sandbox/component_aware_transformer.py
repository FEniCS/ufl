"""."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-22 -- 2009-01-09"

from collections import defaultdict
from itertools import izip, chain

from ufl import *
from ufl.classes import Index, FixedIndex, AxisType
from ufl.common import Stack, StackDict
from ufl.algorithms.transformations import Transformer

class ComponentAwareTransformer(Transformer):
    "..."
    def __init__(self):
        Transformer.__init__(self)
        
        # current indexing status
        self._components = Stack()
        self._index2value = StackDict()
    
    def component(self):
        "Return current component tuple."
        if len(self._components):
            return self._components.peek()
        return ()
    
    def index_value(self, idx):
        "Return value of index if available, otherwise the index itself."
        return self._index2value.get(idx, idx)
    
    ### Index handling: 
    
    def indexed(self, x):
        c = self.component()
        A, ii = x.operands()
        
        xsh = x.shape()
        
        # FIXME: c <- xsh <- ii
        #ii
        
        fi = x.free_indices()
        ri = x.repeated_indices()
        
        # analyse index tuple
        compcount = 0
        subcomp = []
        ri_pos = defaultdict(tuple)
        for k, i in enumerate(ii._indices):
            if isinstance(i, AxisType):
                subcomp.append(c[compcount])
                compcount += 1
            elif isinstance(i, FixedIndex):
                subcomp.append(i._value)
            elif isinstance(i, Index):
                if i in ri:
                    subcomp.append(0)
                    ri_pos[i] += (k,)
                else:
                    subcomp.append(self._index2value[i])
            else:
                ufl_error("Invalid index type %s." % type(i))
        
        
        # no repeated indices makes this simple
        self._components.push(tuple(subcomp))
        result = self.visit(A)
        self._components.pop()
    
    def indexed(self, x):
        
        A, ii = x.operands()
        
        xsh = x.shape()
        c = self.component()
        # FIXME: c <- xsh <- ii
        
        sh = A.shape()
        fi = x.free_indices()
        ri = x.repeated_indices()
        
        # analyse index tuple
        compcount = 0
        subcomp = []
        ri_pos = defaultdict(tuple)
        for k, i in enumerate(ii._indices):
            if isinstance(i, AxisType):
                subcomp.append(c[compcount])
                compcount += 1
            elif isinstance(i, FixedIndex):
                subcomp.append(i._value)
            elif isinstance(i, Index):
                if i in ri:
                    subcomp.append(0)
                    ri_pos[i] += (k,)
                else:
                    subcomp.append(self._index2value[i])
            else:
                ufl_error("Invalid index type %s." % type(i))
        
        # handle implicit sums over repeated indices
        if ri:
            ri = tuple(ri)
            if len(ri) > 1:
                ufl_error("TODO: Multiple repeated indices not implemented yet.") # TODO: Implement to allow A[i,i,j,j], but for now, note that A[i,i,j,j] == A[i,i,:,:][j,j]
            
            result = swiginac.numeric(0)
            
            i, = ri # TODO: only one! need permutations for more
            
            pos = ri_pos[i]
            dim = sh[pos[0]]
            ufl_assert(all(dim == sh[k] for k in pos),
                "Dimension mismatch in implicit sum over Indexed object.")
            
            # for jj in permutations
            for j in range(dim):
                # for ii in ri:
                #    for k in ri_pos[ii]:
                #        subcomp[k] = jj[...]
                for k in ri_pos[i]:
                    subcomp[k] = j
                self._components.push(tuple(subcomp))
                result += self.visit(A)
                self._components.pop()
        else:
            # no repeated indices makes this simple
            self._components.push(tuple(subcomp))
            result = self.visit(A)
            self._components.pop()
        
        return result
    
    ### Container handling:
    
    def list_tensor(self, x):
        component = self.component()
        ufl_assert(len(component) > 0 and \
                   all(isinstance(i, int) for i in component),
                   "Can't index tensor with %s." % repr(component))
        
        # Hide indexing when evaluating subexpression
        self._components.push(())
        
        # Get scalar UFL subexpression from tensor
        e = x
        for i in component:
            e = e._expressions[i]
        ufl_assert(e.shape() == (), "Expecting scalar expression "\
                   "after extracting component from tensor.")
        
        # Apply conversion to scalar subexpression
        r = self.visit(e)
        
        # Return to previous component state
        self._components.pop()
        return r
    
    def component_tensor(self, x):
        # this function evaluates the tensor expression
        # with indices equal to the current component tuple
        expression, indices = x.operands()
        ufl_assert(expression.shape() == (), "Expecting scalar base expression.")
        
        # update index map with component tuple values
        comp = self.component()
        ufl_assert(len(indices) == len(comp), "Index/component mismatch.")
        for i, v in izip(indices._indices, comp):
            self._index2value.push(i, v)
        self._components.push(())
        
        # evaluate with these indices
        result = self.visit(expression)
        
        # revert index map
        for i in range(len(comp)):
            self._index2value.pop()
        self._components.pop()
        return result


class UnusedComponentAwareHandlers:
    
    ### Fallback handlers:
    
    def expr(self, x):
        ufl_error("Missing handler for type %s" % str(type(x)))
    
    def terminal(self, x):
        ufl_error("Missing handler for terminal type %s" % str(type(x)))
    
    ### Handlers for terminal objects:
    
    def zero(self, x):
        ufl_assert(len(self.component()) == len(x.shape()), \
            "Index component length mismatch in zero tensor!")
        TODO
    
    def scalar_value(self, x):
        ufl_assert(self.component() == (),
            "Shouldn't have any component at this point.")
        TODO
    
    def basis_function(self, x):
        c = self.component()
        TODO
    
    def function(self, x):
        c = self.component()
        TODO
    
    def facet_normal(self, x):
        c, = self.component()
        TODO
    
    def spatial_coordinate(self, x):
        c, = self.component()
        TODO
    
    ### Handler for variables:
    
    def variable(self, x):
        c = self.component()
        e, l = x.operands()
        index_values = tuple(self.index_value(k) for k in e.free_indices())
        TODO
    
    ### Handlers for basic algebra:
    
    def sum(self, x, *ops):
        c = self.component()
        TODO
    
    def product(self, x):
        c = self.component()
        TODO
    
    def division(self, x, a, b):
        c = self.component()
        TODO
    
    def abs(self, x, a):
        c = self.component()
        TODO

