from ufl.utils.str import as_native_str
from ufl.coefficient import Coefficient
from ufl.core.ufl_type import ufl_type
from ufl.constantvalue import as_ufl
from ufl.log import error


@ufl_type(inherit_indices_from_operand=0, is_terminal=True, is_differential=True)
class ExternalOperator(Coefficient):

    # Slots are disabled here because they cause trouble in PyDOLFIN
    # multiple inheritance pattern:
    # __slots__ = as_native_strings(("ufl_operands", "_ufl_function_space", "derivatives", "_ufl_shape"))
    _ufl_noslots_ = True

    def __init__(self, *operands, eval_space, derivatives=None, shape=None, count=None):

        # Check operands
        for i in operands:
            as_ufl(i)

        self.ufl_operands = operands

        Coefficient.__init__(self, eval_space, count)
        self._ufl_shape = ()

        # Checks
        if derivatives is not None:
            if not isinstance(derivatives, tuple):
                error("Expecting a tuple for derivatives and not %s" % derivatives)
            if len(derivatives) == len(self.ufl_operands):
                self.derivatives = derivatives
            else:
                error("Expecting a size of %s for %s" % (len(self.ufl_operands), derivatives))
        else:
            self.derivatives = (0,) * len(self.ufl_operands)
        if shape is not None:
            if isinstance(shape, tuple):
                self._ufl_shape = shape
            else:
                error("Expecting a tuple for the shape and not %s" % shape)

    def evaluate(self, x, mapping, component, index_values):
        """Evaluate expression at given coordinate with given values for terminals."""
        error("Symbolic evaluation of %s not available." % self._ufl_class_.__name__)

    def _ufl_expr_reconstruct_(self, *operands, eval_space=None, derivatives=None, shape=None, count=None):
        "Return a new object of the same type with new operands."
        return self._ufl_class_(*operands, eval_space=eval_space, derivatives=derivatives, shape=shape, count=count)

    def __str__(self):
        "Default repr string construction for ExternalOperator operators."
        # This should work for most cases
        r = "%s(%s,%s,%s,%s,%s)" % (self._ufl_class_.__name__, ", ".join(repr(op) for op in self.ufl_operands), repr(self._ufl_function_space), repr(self.derivatives), repr(self._ufl_shape), repr(self._count))
        return as_native_str(r)
