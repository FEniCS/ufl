from ufl.utils.str import as_native_str
from ufl.utils.str import as_native_strings
from ufl.core.operator import Operator
from ufl.core.ufl_type import ufl_type
from ufl.finiteelement import FiniteElementBase
from ufl.functionspace import AbstractFunctionSpace
from ufl.constantvalue import as_ufl
from ufl.log import error


@ufl_type(inherit_indices_from_operand=0, num_ops=1, is_terminal_modifier=True, is_differential=True)
class ExternalOperator(Operator):
    __slots__ = as_native_strings(("ufl_operands", "eval_space", "deriv_index", "ufl_shape"))

    def __init__(self, *operands, eval_space=None, derivatives=None, shape=None):

        # Check operands
        for i in operands:
            as_ufl(i)

        Operator.__init__(self, operands)
        self.ufl_shape = ()

        # Checks
        if eval_space is not None:
            if isinstance(eval_space, AbstractFunctionSpace) or isinstance(eval_space, FiniteElementBase):
                self.eval_space = eval_space
            else:
                error("Expecting a FunctionSpace or FiniteElement")
        if derivatives is not None:
            if not isinstance(derivatives, tuple):
                error("Expecting a tuple for derivatives and not %s" % derivatives)
            if len(derivatives) == len(self.ufl_operands):
                self.deriv_index = derivatives
            else:
                error("Expecting a size of %s for %s" % (len(self.ufl_operands), derivatives))
        else:
            self.deriv_index = (0,) * len(self.ufl_operands)
        if shape is not None:
            if isinstance(shape, tuple):
                self.ufl_shape = shape
            else:
                error("Expecting a tuple for the shape and not %s" % shape)

    def evaluate(self, x, mapping, component, index_values):
        """Evaluate expression at given coordinate with given values for terminals."""
        error("Symbolic evaluation of %s not available." % self._ufl_class_.__name__)

    def __str__(self):
        "Default repr string construction for ExternalOperator operators."
        # This should work for most cases
        space = None
        if hasattr(self, 'eval_space'):
            space = self.eval_space
        r = "%s(%s,%s,%s,%s)" % (self._ufl_class_.__name__, ", ".join(repr(op) for op in self.ufl_operands), repr(space), repr(self.deriv_index), repr(self.ufl_shape))
        return as_native_str(r)
