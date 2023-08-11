"""
This module defines the ``ExternalOperator`` class, which symbolically represents operators that
are not straightforwardly expressible in UFL. A practical implementation is required at a later
stage to define how this operator should be evaluated as well as its derivatives from a given set
of operands.
"""

# Copyright (C) 2019 Nacime Bouziani
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Nacime Bouziani, 2020

from ufl.coefficient import Coefficient
from ufl.core.operator import Operator
from ufl.core.ufl_type import ufl_type
from ufl.constantvalue import as_ufl
from ufl.finiteelement.finiteelement import FiniteElement
from ufl.finiteelement.mixedelement import VectorElement, TensorElement
from ufl.functionspace import FunctionSpace
from ufl.referencevalue import ReferenceValue


@ufl_type(num_ops="varying", inherit_indices_from_operand=0, is_differential=True)
class ExternalOperator(Operator):

    # Slots are disabled here because they cause trouble in PyDOLFIN
    # multiple inheritance pattern:
    _ufl_noslots_ = True

    def __init__(self, *operands, function_space, derivatives=None, coefficient=None, arguments=(), local_operands=()):
        r"""
        :param operands: operands on which acts the :class:`ExternalOperator`.
        :param function_space: the :class:`.FunctionSpace`,
            or :class:`.MixedFunctionSpace` on which to build this :class:`Function`.
            Alternatively, another :class:`Coefficient` may be passed here and its function space
            will be used.
        :param derivatives: tuple specifiying the derivative multiindex.
        :param coefficient: ufl.Coefficient associated to the ExternalOperator representing what is
            produced by the operator
        :param arguments: tuple composed of tuples whose first argument is a ufl.Argument or ufl.Expr
            containing several ufl.Argument objects and whose second arguments is a boolean indicating
            whether we take the action of the adjoint. We have arguments when the operator is a GlobalExternalOperator.
        :param local_operands: tuple specyfing the operands on which the operator acts locally
        """

        ufl_operands = tuple(map(as_ufl, operands))
        Operator.__init__(self, ufl_operands)

        # Process arguments and action arguments
        arguments = tuple((as_ufl(args), is_adjoint) for args, is_adjoint in arguments)
        self._action_coefficients, self._arguments = self._extract_coeffs_and_args(arguments)

        # Process local operands
        self.local_operands = tuple(map(as_ufl, local_operands))

        # Make the coefficient associated to the external operator
        ref_coefficient = Coefficient(function_space)

        # Checks
        if derivatives is not None:
            if not isinstance(derivatives, tuple):
                raise TypeError("Expecting a tuple for derivatives and not %s" % derivatives)
            if not len(derivatives) == len(self.ufl_operands):
                raise TypeError("Expecting a size of %s for %s" % (len(self.ufl_operands), derivatives))

            self.derivatives = derivatives
            # If we have arguments, the appropriate function space has already been set up upstream
            if len(arguments) == 0:
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
                s = ref_coefficient.ufl_shape
                for i, e in enumerate(self.derivatives):
                    s += self.ufl_operands[i].ufl_shape * e
                new_function_space = self._make_function_space(s, sub_element=ref_coefficient.ufl_element(),
                                                               domain=ref_coefficient.ufl_domain())
            else:
                new_function_space = ref_coefficient.ufl_function_space()
        else:
            new_function_space = ref_coefficient.ufl_function_space()
            self.derivatives = (0,) * len(self.ufl_operands)

        if coefficient is None:
            coefficient = Coefficient(new_function_space)
        elif not isinstance(coefficient, (Coefficient, ReferenceValue)):
            raise TypeError('Expecting a Coefficient and not %s', type(coefficient))
        self._coefficient = coefficient

        if not self.coefficient().ufl_function_space() == new_function_space:
            raise ValueError('The function spaces do not match!')

        if self.derivatives == (0,) * len(self.ufl_operands):
            self._extop_master = self
            self.coefficient_dict = {}

    def coefficient(self):
        "Returns the coefficient produced by the external operator"
        return self._coefficient

    def _extract_coeffs_and_args(self, operands):
        from ufl.algorithms.analysis import extract_arguments
        args = tuple((e, is_adjoint) for e, is_adjoint in operands if len(extract_arguments(e)) != 0)
        ops = tuple((e, is_adjoint) for e, is_adjoint in operands if (e, is_adjoint) not in args)
        return ops, args

    def get_coefficient(self):
        """Helper function returning the coefficient produced by the external operator"""
        if isinstance(self._coefficient, ReferenceValue):
            return self._coefficient.ufl_operands[0]
        else:
            return self._coefficient

    def arguments(self):
        """Returns a tuple of expressions containing an argument.
        This is the case when we take the Gateaux derivative of a GloibalExternalOperator"""
        return self._arguments

    def action_coefficients(self):
        """Returns a tuple of expressions containing a coefficient. When we take the action of a GlobalExternalOperator,
        the arguments in self.arguments() are replaced by coefficients.
        self.action_coefficients() is equivalent to `ufl.replace(self.arguments(),
        dictionary_mapping_arguments_to_coefficients)`"""
        return self._action_coefficients

    @property
    def is_type_global(self):
        "States if the external operator is global"
        local_operands = self.local_operands
        return tuple(e not in local_operands for e in self.ufl_operands)

    def count(self):
        "Returns the count associated to the coefficient produced by the external operator"
        return self._count

    @property
    def _count(self):
        return self.get_coefficient()._count

    @property
    def ufl_shape(self):
        "Returns the UFL shape of the coefficient.produced by the external operator"
        return self.get_coefficient()._ufl_shape

    def ufl_function_space(self):
        "Returns the ufl function space associated to the external operator, the one we interpolate the operands on."
        return self.get_coefficient()._ufl_function_space

    def _make_function_space_args(self, k, y, adjoint=False):
        r"""Make the function space of the Gateaux derivative:
        dN[x] = \\frac{dN}{dOperands[k]} * y(x) if adjoint is False
        and of \\frac{dN}{dOperands[k]}^{*} * y(x) if adjoint is True"""
        opk_shape = self.ufl_operands[k].ufl_shape
        y_shape = y.ufl_shape
        shape = self.ufl_function_space().ufl_element().reference_value_shape()
        for i, e in enumerate(self.derivatives):
            shape += self.ufl_operands[i].ufl_shape * (e - int(i == k))

        if not adjoint:
            add_shape = y_shape[len(opk_shape):]
            shape += add_shape
        else:
            add_shape = y_shape[len(shape):]
            shape = tuple(reversed(opk_shape)) + add_shape
        return self._make_function_space(shape)

    def _make_function_space(self, s, sub_element=None, domain=None):
        """Make the function space of a Coefficient of shape s"""
        if sub_element is None:
            sub_element = self.ufl_function_space().ufl_element()
        if domain is None:
            domain = self.ufl_function_space().ufl_domain()
        if not isinstance(sub_element, (FiniteElement, VectorElement, TensorElement)):
            # While TensorElement (resp. VectorElement) allows to build a tensor element of a given shape
            # where all the elements in the tensor (resp. vector) are equal, there is no mechanism to construct
            # an element of a given shape with hybrid elements in it.
            # MixedElement flatten out the shape of the elements passed in as arguments.
            # For instance, starting from an ExternalOperator based on an MixedElement (F1, F2) of shape (2,),
            # we would like to have a mechanism to construct the gradient of it based on the
            # MixedElement (F1, F1)
            #              (F2, F2) of shape (2, 2)
            # TODO: subclass MixedElement to be able to do that !
            raise NotImplementedError("MixedFunctionSpaces not handled yet")

        if len(sub_element.sub_elements()) != 0:
            sub_element = sub_element.sub_elements()[0]

        if len(s) == 0:
            ufl_element = sub_element
        elif len(s) == 1:
            ufl_element = VectorElement(sub_element, dim=s[0])
        else:
            ufl_element = TensorElement(sub_element, shape=s)
        return FunctionSpace(domain, ufl_element)

    def _grad(self):
        """Returns the symbolic grad of the external operator"""
        # By default, differential rules produce grad(o.get_coefficient()) since
        # the external operator may not be smooth enough for chain rule to hold.
        # Symbolic gradient (grad(ExternalOperator)) depends on the operator considered
        # and its implementation may be needed in some cases (e.g. convolution operator).
        raise NotImplementedError('Symbolic gradient not defined for the external operator considered!')

    def evaluate(self, x, mapping, component, index_values):
        """Evaluate expression at given coordinate with given values for terminals."""
        raise TypeError("Symbolic evaluation of %s not available." % self._ufl_class_.__name__)

    def _ufl_expr_reconstruct_(
        self, *operands, function_space=None, derivatives=None, coefficient=None,
        arguments=None, local_operands=None, add_kwargs={}
    ):
        "Return a new object of the same type with new operands."
        deriv_multiindex = derivatives or self.derivatives

        if deriv_multiindex != self.derivatives:
            # If we are constructing a derivative
            corresponding_coefficient = None
            e_master = self._extop_master
            for ext in e_master.coefficient_dict.values():
                if ext.derivatives == deriv_multiindex:
                    return ext._ufl_expr_reconstruct_(*operands, function_space=function_space,
                                                      derivatives=deriv_multiindex,
                                                      coefficient=coefficient,
                                                      arguments=arguments,
                                                      local_operands=local_operands,
                                                      add_kwargs=add_kwargs)
        else:
            corresponding_coefficient = coefficient or self._coefficient

        reconstruct_op = type(self)(*operands, function_space=function_space or self.ufl_function_space(),
                                    derivatives=deriv_multiindex,
                                    coefficient=corresponding_coefficient,
                                    arguments=arguments or (self.arguments() + self.action_coefficients()),
                                    local_operands=local_operands or self.local_operands,
                                    **add_kwargs)

        if deriv_multiindex != self.derivatives:
            # If we are constructing a derivative
            self._extop_master.coefficient_dict.update({deriv_multiindex: reconstruct_op})
            reconstruct_op._extop_master = self._extop_master
        else:
            reconstruct_op._extop_master = self._extop_master
        return reconstruct_op

    def __repr__(self):
        "Default repr string construction for operators."
        # This should work for most cases
        r = f"ExternalOperator({', '.join(repr(op) for op in self.ufl_operands)}; {self._arguments}, {self._count})"
        return r

    def __str__(self):
        "Default repr string construction for ExternalOperator operators."
        # This should work for most cases
        r = "%s(%s,%s,%s,%s,%s)" % (self._ufl_class_.__name__, ", ".join(repr(op) for op in self.ufl_operands),
                                    repr(self.ufl_function_space()), repr(self.derivatives), repr(self.ufl_shape),
                                    repr(self.count()))
        return r

    def _ufl_compute_hash_(self):
        "Default hash of terminals just hash the repr string."
        return hash(repr(self))

    def _ufl_signature_data_(self, renumbering):
        "Signature data for form arguments depend on the global numbering of the form arguments and domains."
        coefficient_signature = self.get_coefficient()._ufl_signature_data_(renumbering)
        return ("ExternalOperator", *self.is_type_global, *coefficient_signature, *self.derivatives)

    def __eq__(self, other):
        if not isinstance(other, ExternalOperator):
            return False
        if self is other:
            return True
        return (self.count() == other.count() and
                self.ufl_function_space() == other.ufl_function_space())
