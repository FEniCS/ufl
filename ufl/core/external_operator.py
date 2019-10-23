from ufl.coefficient import Coefficient
from ufl.core.ufl_type import ufl_type
from ufl.constantvalue import as_ufl
from ufl.log import error
from ufl.finiteelement.finiteelement import FiniteElement
from ufl.finiteelement.mixedelement import VectorElement, TensorElement
from ufl.functionspace import FunctionSpace
from ufl.domain import default_domain


@ufl_type(inherit_indices_from_operand=0, is_terminal=True, is_differential=True)
class ExternalOperator(Coefficient):

    # Slots are disabled here because they cause trouble in PyDOLFIN
    # multiple inheritance pattern:
    _ufl_noslots_ = True

    def __init__(self, *operands, function_space, derivatives=None, count=None, extop_id=None):
        r"""
        :param operands: operands on which acts the :class:`ExternalOperator`.
        :param function_space: the :class:`.FunctionSpace`,
        or :class:`.MixedFunctionSpace` on which to build this :class:`Function`.
        Alternatively, another :class:`Function` may be passed here and its function space
        will be used to build this :class:`Function`.  In this case, the function values are copied.
        :param derivatives: tuple scecifiying the derivative multiindex.
        :param extop_id: dictionary that store the position of the :class:`ExternalOperator` in the forms where it turns up.
        """

        self.ufl_operands = tuple(map(as_ufl, operands))
        Coefficient.__init__(self, function_space, count=count)

        # Checks
        if derivatives is not None:
            if not isinstance(derivatives, tuple):
                error("Expecting a tuple for derivatives and not %s" % derivatives)
            if len(derivatives) == len(self.ufl_operands):
                # In this block, we construct the function space on which lives the ExternalOperator
                # accordingly to the derivatives taken, for instance in 2d:
                #      for V = VectorFunctionSpace(...)
                #      if u = Function(V); e = ExternalOperator(u, function_space=V)
                #      then, e.ufl_shape = (2,)
                #            dedu.ufl_shape = (2,2)
                #            ...
                # Therefore, for 'dedu' we need to construct the new TensorElement to end up with the appropriate
                # function space and we also need to store the VectorElement corresponding to V since it is on this
                # function_space that we will interpolate the operands.
                self.derivatives = derivatives
                s = self.ufl_shape
                for i, e in enumerate(self.derivatives):
                    s += self.ufl_operands[i].ufl_shape * e
                if len(s) > len(self.ufl_shape):
                    sub_element = self.ufl_element()
                    if isinstance(sub_element, (FiniteElement, VectorElement, TensorElement)):
                        if len(sub_element.sub_elements()) != 0:
                            sub_element = sub_element.sub_elements()[0]
                        ufl_element = TensorElement(sub_element, shape=s)
                        domain = default_domain(ufl_element.cell())
                        self._original_function_space = FunctionSpace(domain, ufl_element)
                        self._ufl_shape = ufl_element.reference_value_shape()
                    else:
                        # While TensorElement (resp. VectorElement) allows to build a tensor element of a given shape
                        # where all the elements in the tensor (resp. vector) are equal, there is no mechanism to construct
                        # an element of a given shape with hybrid elements in it.
                        # MixedElement flatten out the shape of the elements passed in as arguments.
                        # For instance, starting from an ExternalOperator based on an MixedElement (F1, F2) of shape (2,),
                        # we would like to have a mechanism to construct the gradient of it based on the
                        # MixedElement (F1, F1)
                        #              (F2, F2) of shape (2, 2)
                        # TODO: subclass MixedElement to be able to do that !
                        error("MixedFunctionSpaces not handled yet")
                else:
                    self._ufl_shape = self.ufl_element().reference_value_shape()
                    self._original_function_space = self._ufl_function_space
            else:
                error("Expecting a size of %s for %s" % (len(self.ufl_operands), derivatives))
        else:
            self._ufl_shape = self.ufl_element().reference_value_shape()
            self._original_function_space = self._ufl_function_space
            self.derivatives = (0,) * len(self.ufl_operands)

        self._extop_dependencies = [self, ]

        if self.derivatives == (0,) * len(self.ufl_operands):
            self._extop_master = self

    def original_function_space(self):
        return self._original_function_space

    def evaluate(self, x, mapping, component, index_values):
        """Evaluate expression at given coordinate with given values for terminals."""
        error("Symbolic evaluation of %s not available." % self._ufl_class_.__name__)

    def _ufl_expr_reconstruct_(self, *operands, function_space=None, derivatives=None, count=None):
        "Return a new object of the same type with new operands."
        deriv_multiindex = derivatives or self.derivatives

        if deriv_multiindex != self.derivatives:
            # If we are constructing a derivative
            corresponding_count = None
            e_master = self._extop_master
            for ext in e_master._extop_dependencies:
                if ext.derivatives == deriv_multiindex:
                    return ext._ufl_expr_reconstruct_(*operands, function_space=function_space,
                                                      derivatives=deriv_multiindex, count=count)
        else:
            corresponding_count = self._count

        if deriv_multiindex != self.derivatives:
            corresponding_count = None
        else:
            corresponding_count = self._count

        reconstruct_op = type(self)(*operands, function_space=function_space or self._ufl_function_space,
                                    derivatives=deriv_multiindex,
                                    count=corresponding_count)

        if deriv_multiindex != self.derivatives:
            # If we are constructing a derivative
            self._extop_master._extop_dependencies.append(reconstruct_op)
            reconstruct_op._extop_master = self._extop_master
        else:
            reconstruct_op._extop_master = self._extop_master
            reconstruct_op._extop_dependencies = self._extop_dependencies
        return reconstruct_op

    def _add_dependencies(self, args):
        v = list(self._ufl_expr_reconstruct_(*self.ufl_operands, derivatives=o) for o in args)
        self._extop_master._extop_dependencies.extend(x for x in v if x not in self._extop_master._extop_dependencies)
        return self

    def __str__(self):
        "Default repr string construction for ExternalOperator operators."
        # This should work for most cases
        r = "%s(%s,%s,%s,%s,%s)" % (self._ufl_class_.__name__, ", ".join(repr(op) for op in self.ufl_operands), repr(self._ufl_function_space), repr(self.derivatives), repr(self._ufl_shape), repr(self._count))
        return r
