from ufl.utils.str import as_native_str
from ufl.coefficient import Coefficient
from ufl.core.ufl_type import ufl_type
from ufl.constantvalue import as_ufl
from ufl.log import error
from ufl.finiteelement.mixedelement import TensorElement
from ufl.functionspace import FunctionSpace
from ufl.domain import as_domain, default_domain
import weakref


@ufl_type(inherit_indices_from_operand=0, is_terminal=True, is_differential=True)
class ExternalOperator(Coefficient):

    # Slots are disabled here because they cause trouble in PyDOLFIN
    # multiple inheritance pattern:
    # __slots__ = as_native_strings(("ufl_operands", "_ufl_function_space", "derivatives", "_ufl_shape"))
    _ufl_noslots_ = True
    _ufl_all_external_operators_ = weakref.WeakKeyDictionary()

    def __init__(self, *operands, function_space, derivatives=None, count=None):

        # Check operands
        for i in operands:
            as_ufl(i)

        self.ufl_operands = operands

        Coefficient.__init__(self, function_space, count)

        # Checks
        if derivatives is not None:
            if not isinstance(derivatives, tuple):
                error("Expecting a tuple for derivatives and not %s" % derivatives)
            if len(derivatives) == len(self.ufl_operands):
                self.derivatives = derivatives
                s = self.ufl_shape
                for i, e in enumerate(self.derivatives):
                    s += self.ufl_operands[i].ufl_shape*e
                if len(s) > len(self.ufl_shape):
                    sub_element = self._ufl_function_space.ufl_element()
                    if len(sub_element.sub_elements()) != 0:
                        sub_element = sub_element.sub_elements()[0]
                    ufl_element = TensorElement(sub_element, shape=s)
                    domain = default_domain(ufl_element.cell())
                    self.original_function_space = FunctionSpace(domain, ufl_element)
                    self._ufl_shape = ufl_element.reference_value_shape()
                else:
                    self._ufl_shape = self._ufl_function_space.ufl_element().reference_value_shape()
                    self.original_function_space = self._ufl_function_space
            else:
                error("Expecting a size of %s for %s" % (len(self.ufl_operands), derivatives))
        else:
            self._ufl_shape = self._ufl_function_space.ufl_element().reference_value_shape()
            self.original_function_space = self._ufl_function_space
            self.derivatives = (0,) * len(self.ufl_operands)

        key_e = find_initial_external_operator(self)
        if key_e is None:#not self.name() in tuple(e.name() for e in type(self)._ufl_all_external_operators_.keys()):
            type(self)._ufl_all_external_operators_[self] = weakref.WeakValueDictionary()
            type(self)._ufl_all_external_operators_[self][self.derivatives] = self


    def evaluate(self, x, mapping, component, index_values):
        """Evaluate expression at given coordinate with given values for terminals."""
        error("Symbolic evaluation of %s not available." % self._ufl_class_.__name__)

    def _ufl_expr_reconstruct_(self, *operands, function_space=None, derivatives=None, count=None):
        "Return a new object of the same type with new operands."
        deriv_multiindex = derivatives or self.derivatives

        if deriv_multiindex in type(self)._ufl_all_external_operators_[self].keys():
            corresponding_count = type(self)._ufl_all_external_operators_[self][deriv_multiindex]._count
            self._ufl_class_(*operands, function_space=function_space or self.original_function_space,#change by intrinsic space
                                derivatives=deriv_multiindex,
                                count=corresponding_count)
        else:
            reconstruct_extop = self._ufl_class_(*operands, function_space=function_space or self.original_function_space,#change by intrinsic space
                                derivatives=deriv_multiindex)

            type(self)._ufl_all_external_operators_[self][deriv_multiindex] = reconstruct_extop
            return reconstruct_extop
    def __str__(self):
        "Default repr string construction for ExternalOperator operators."
        # This should work for most cases
        r = "%s(%s,%s,%s,%s,%s)" % (self._ufl_class_.__name__, ", ".join(repr(op) for op in self.ufl_operands), repr(self._ufl_function_space), repr(self.derivatives), repr(self._ufl_shape), repr(self._count))
        return as_native_str(r)


def find_initial_external_operator(eop):
    k = None
    for key, val in ExternalOperator._ufl_all_external_operators_.items():
        for e in val.values():
            if e._count == eop._count:
                k = key
    return k