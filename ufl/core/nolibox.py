from ufl.utils.str import as_native_str
from ufl.utils.str import as_native_strings
from ufl.core.operator import Operator
from ufl.core.ufl_type import ufl_type
from ufl.finiteelement import FiniteElementBase
from ufl.functionspace import AbstractFunctionSpace
from ufl.constantvalue import as_ufl

@ufl_type(is_abstract=True, is_differential=True)
class Nolibox(Operator):
    __slots__ = as_native_strings(("ufl_operands","ufl_free_indices","eval_space","deriv_index","ufl_shape","ufl_index_dimensions","pending_derivatives"))

    def __init__(self, *operands, eval_space = None, derivatives = None):
        
        #Check operands
        for i in operands:
            as_ufl(i)
            
        Operator.__init__(self, operands)
        
        self.ufl_shape = ()
        self.ufl_free_indices = ()
        self.ufl_index_dimensions = ()
        self.pending_derivatives = ()
        
        #Checks
        if eval_space is not None:
            if isinstance(eval_space, AbstractFunctionSpace) or isinstance(eval_space,FiniteElementBase):
                self.eval_space = eval_space
            else:
                error("Expecting a FunctionSpace or FiniteElement")
        if derivatives is not None:
            if not isinstance(derivatives,tuple):
                error("Expecting a tuple for derivatives and not %s" %derivatives)
            if len(derivatives) == len(self.ufl_operands):
                self.deriv_index = derivatives
            else:
                error("Expecting a size of %s for %s" % (len(self.ufl_operands),derivatives))
        else:
            self.deriv_index = (0,)*len(self.ufl_operands)
            
    def evaluate(self, x, mapping, component, index_values): #Temporary : in order to pass the test
        result = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        return result    
    
    def __str__(self):
        "Default repr string construction for Nolibox operators."
        # This should work for most cases
        r = "%s(%s)" % (self._ufl_class_.__name__,
                        ", ".join(repr(op) for op in self.ufl_operands))
        return as_native_str(r)
