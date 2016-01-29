#!/usr/bin/env py.test
# -*- coding: utf-8 -*-
"""Tests of the change to reference frame algorithm."""

import pytest

from ufl import *

from ufl.classes import Form, Integral, Expr, ReferenceGrad, ReferenceValue

'''
from ufl.classes import ReferenceGrad, JacobianInverse
from ufl.algorithms import tree_format, change_to_reference_grad

from six.moves import xrange as range

from ufl.log import error, warning
from ufl.assertions import ufl_assert

from ufl.core.multiindex import Index, indices
from ufl.corealg.multifunction import MultiFunction
from ufl.corealg.map_dag import map_expr_dag

from ufl.classes import (Expr, FormArgument, GeometricQuantity,
                         Terminal, ReferenceGrad, Grad, Restricted, ReferenceValue,
                         Jacobian, JacobianInverse, JacobianDeterminant,
                         FacetJacobian, FacetJacobianInverse, FacetJacobianDeterminant,
                         CellFacetJacobian,
                         CellEdgeVectors, FacetEdgeVectors,
                         FacetNormal, CellNormal, ReferenceNormal,
                         CellVolume, FacetArea,
                         CellOrientation, FacetOrientation, QuadratureWeight,
                         SpatialCoordinate, Indexed, MultiIndex, FixedIndex)

from ufl.constantvalue import as_ufl, Identity
from ufl.tensoralgebra import Transposed
from ufl.tensors import as_tensor, as_vector, as_scalar, ComponentTensor
from ufl.operators import sqrt, max_value, min_value, sign
from ufl.permutation import compute_indices

from ufl.algorithms.transformer import ReuseTransformer, apply_transformer
from ufl.compound_expressions import determinant_expr, cross_expr, inverse_expr
from ufl.finiteelement import FiniteElement, EnrichedElement, VectorElement, MixedElement, TensorProductElement, TensorElement, FacetElement, InteriorElement, BrokenElement, TraceElement
'''


def change_integral_to_reference_frame(form, context):
    if False: # TODO: integral.is_in_reference_frame():
        # TODO: Assume reference frame integral is written purely in
        #       reference frame or tramsform integrand here as well?
        return integrand
    else:
        # Change integrand expression to reference frame
        integrand = change_to_reference_frame(integral.integrand())

        # Compute and apply integration scaling factor
        scale = compute_integrand_scaling_factor(integral.ufl_domain(),
                                                 integral.integral_type())

        return integral.reconstruct(integrand * scale) # TODO: , reference=True)


def change_expr_to_reference_frame(expr):
    expr = ReferenceValue(expr)
    return expr


def change_to_reference_frame(expr):
    if isinstance(expr, Form):
        return change_form_to_reference_frame(expr)
    elif isinstance(expr, Integral):
        return change_integral_to_reference_frame(expr)
    elif isinstance(expr, Expr):
        return change_expr_to_reference_frame(expr)
    else:
        error("Invalid type.")


def test_change_unmapped_form_arguments_to_reference_frame():
    U = FiniteElement("CG", triangle, 1)
    V = VectorElement("CG", triangle, 1)
    T = TensorElement("CG", triangle, 1)

    expr = Coefficient(U)
    assert change_to_reference_frame(expr) == ReferenceValue(expr)
    expr = Coefficient(V)
    assert change_to_reference_frame(expr) == ReferenceValue(expr)
    expr = Coefficient(T)
    assert change_to_reference_frame(expr) == ReferenceValue(expr)


def test_change_hdiv_form_arguments_to_reference_frame():
    V = FiniteElement("RT", triangle, 1)

    expr = Coefficient(V)
    assert change_to_reference_frame(expr) == ReferenceValue(expr)


def test_change_hcurl_form_arguments_to_reference_frame():
    V = FiniteElement("RT", triangle, 1)

    expr = Coefficient(V)
    assert change_to_reference_frame(expr) == ReferenceValue(expr)

    '''
    # user input
    grad(f + g)('+')
    # change to reference frame
    -> (K*rgrad(rv(M)*rv(f) + rv(M)*rv(g)))('+')
    # apply derivatives
    -> (K*(rv(M)*rgrad(rv(f)) + rv(M)*rgrad(rv(g))))('+')
    # apply restrictions
    -> K('+')*(rv(M('+'))*rgrad(rv(f('+'))) + rv(M('+'))*rgrad(rv(g('+'))))



    # user input
    grad(f + g)('+')

    # some derivatives applied before processing
    (grad(f) + grad(g))('+')

    # ... replace to get fully defined form arguments here ...

    # expand compounds
    # * context options:
    #   - keep {types} without rewriting to lower level types
    #   - preserve div and curl if applied directly to terminal
    #     (ffc context may set this to off)
    # * output invariants:
    #   - no compound operator types left in expression (simplified language)
    #   - div and curl rewritten in terms of grad (optionally unless applied directly to terminal)
    -> (grad(f) + grad(g))('+')

    # change to reference frame
    # * context options:
    #   - keep {types} without rewriting to lower level types (e.g. JacobianInverse)
    #     (ffc context may initially add all code snippets expressions)
    #   - keep {types} in global frame (e.g. Coefficient)
    #     (ffc context may initially add Coefficient and Argument here to refrain from changing)
    #   - skip integral scaling
    #     (ffc context may turn skipping on to preserve current behaviour)
    # * output invariants:
    #   - ReferenceValue bound directly to terminals where applicable
    #   - grad replaced by mapping expression of rgrad
    #   - div replaced by mapping expression of rdiv
    #   - curl replaced by mapping expression of rcurl
    -> as_tensor(IndexSum(K[i,j]*rgrad(as_tensor(rv(M)[k,l]*rv(f)[l], (l,))
                                     + as_tensor(rv(M)[r,s]*rv(g)[s], (s,)))[j],
                          j),
                 (i,))('+')

    # apply derivatives
    # * context options:
    #   - N/A?
    # * output invariants:
    #   - grad,div,curl, bound directly to terminals
    #   - rgrad,rdiv,rcurl bound directly to referencevalue objects (rgrad(global_f) invalid)
    -> (K*(rv(M)*rgrad(rv(f)) + rv(M)*rgrad(rv(g))))('+')

    # apply restrictions
    # * context options:
    #   - N/A?
    # * output invariants:
    #   - *_restricted bound directly to terminals
    #   - all terminals that must be restricted to make sense are restricted
    -> K('+')*(rv(M('+'))*rgrad(rv(f('+'))) + rv(M('+'))*rgrad(rv(g('+'))))

    # final modified terminal structure:
    t = terminal | restricted(terminal)  # choice of terminal
    r = rval(t) | rgrad(r)               # in reference frame: value or n-gradient
    g = t | grad(g)                      # in global frame: value or n-gradient
    v = r | g                            # value in either frame
    e = v | cell_avg(v) | facet_avg(v) | at_cell_midpoint(v) | at_facet_midpoint(v)
                                         # evaluated at point or averaged over cell entity
    m = e | indexed(e)                   # scalar component of
    '''

def new_analyse_modified_terminal(expr):
    assert expr._ufl_is_terminal_ or expr._ufl_is_terminal_modifier_type_
    m = expr

    # The outermost expression may index to get a specific scalar value
    if isinstance(m, Indexed):
        unindexed, multi_index = m.ufl_operands
        indices = tuple(int(i) for i in multi_index)
    else:
        unindexed = m
        indices = ()

    # The evaluation mode is one of current point,
    # a cell entity midpoint, or averaging over a cell entity
    if unindexed._ufl_is_evaluation_type_: # averages and point evaluations
        v, = v.ufl_operand
        evaluation = unindexed.ufl_handler_name
    else:
        v = unindexed
        evaluation = "current_point"

    # We're either in reference frame or global, checks below ensure we don't mix the two
    frame = "reference" if v._ufl_is_reference_type_ else "global"

    # Peel off derivatives (grad^n,div,curl,div(grad),grad(div) etc.)
    t = v
    derivatives = []
    while t._ufl_is_derivative_type_:
        # ensure frame consistency
        assert t._ufl_is_reference_type_ == v._ufl_is_reference_type_
        derivatives.append(t._ufl_class_)
        t, = t.ufl_operands
    core = t
    derivatives = tuple(derivatives)

    # This can be an intermediate step to use derivatives instead of ngrads:
    num_derivatives = len(derivatives)

    # If we had a reference type before unwrapping terminal,
    # there should be a ReferenceValue inside all the derivatives
    if v._ufl_is_reference_type_:
        assert isinstance(t, ReferenceValue)
        t, = t.ufl_operands

    # At the core we may have a restriction
    if t._ufl_is_restriction_type_:
        restriction = t.side()
        t, = t.ufl_operands
    else:
        restriction = ""

    # And then finally the terminal
    assert t._ufl_is_terminal_
    terminal = t

    # This will only be correct for derivatives = grad^n
    gdim = terminal.ufl_domain().geometric_dimension()
    derivatives_shape = (gdim,)*num_derivatives

    # Get shapes
    expr_shape = expr.ufl_shape
    unindexed_shape = unindexed.ufl_shape
    core_shape = core.ufl_shape

    # Split indices
    core_indices = indices[:len(core_shape)]
    derivative_indices = indices[len(core_shape):]

    # Apply paranoid dimension checking
    assert len(indices) == len(unindexed_shape)
    assert all(0 <= i for i in indices)
    assert all(i < j for i,j in zip(indices, unindexed_shape))
    assert len(core_indices) == len(core_shape)
    assert all(0 <= i for i in core_indices)
    assert all(i < j for i,j in zip(core_indices, core_shape))
    assert len(derivative_indices) == len(derivatives_shape) # This will fail for e.g. div(grad(f))
    assert all(0 <= i for i in derivative_indices)
    assert all(i < j for i,j in zip(derivative_indices, derivatives_shape))

    # Return values:
    mt = ModifiedTerminal(
        # TODO: Use keyword args
        expr,
        indices,
        evaluation,
        frame,
        num_derivatives,
        derivatives,
        restriction,
        terminal
        )
    return mt


'''
New form preprocessing pipeline:

Preferably introduce these changes:
1) Create new FormArgument Expression without element or domain
2) Create new FormArgument Constant without domain
3) Drop replace
--> but just applying replace first is fine

i) group and join integrals by (domain, type, subdomain_id),
ii) process integrands:
    a) apply_coefficient_completion # replace coefficients to ensure proper elements and domains
    b) lower_compound_operators # expand_compounds
    c) change_to_reference_frame # change f->rv(f), m->M*rv(m), grad(f)->K*rgrad(rv(f)), grad(grad(f))->K*rgrad(K*rgrad(rv(f))), grad(expr)->K*rgrad(expr)
                                 # if grad(expr)->K*rgrad(expr) should be valid, then rgrad must be applicable to quite generic expressions
    d) apply_derivatives         # one possibility is to add an apply_mapped_derivatives AD algorithm which includes mappings
    e) apply_geometry_lowering
    f) apply_restrictions # requiring grad(f)('+') instead of grad(f('+')) would simplify a lot...
iii) extract final metadata about elements and coefficient ordering
'''
