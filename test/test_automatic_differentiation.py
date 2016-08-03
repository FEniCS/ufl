#!/usr/bin/env py.test
# -*- coding: utf-8 -*-

"""
These tests should cover the behaviour of the automatic differentiation
algorithm at a technical level, and are thus implementation specific.
Other tests check for mathematical correctness of diff and derivative.
"""

from __future__ import division, absolute_import, print_function, unicode_literals

import pytest
from itertools import chain

import ufl

# This imports everything external code will see from ufl
from ufl import *

import ufl.algorithms
from ufl.corealg.traversal import unique_post_traversal
from ufl.conditional import Conditional
from ufl.algorithms import expand_derivatives


class ExpressionCollection(object):

    def __init__(self, cell):
        self.cell = cell

        d = cell.geometric_dimension()
        x = SpatialCoordinate(cell)
        n = FacetNormal(cell)
        c = CellVolume(cell)
        h = Circumradius(cell)
        f = FacetArea(cell)
        #s = CellSurfaceArea(cell)
        # FIXME: Add all new geometry types here!

        I = Identity(d)
        eps = PermutationSymbol(d)

        U = FiniteElement("U", cell, None)
        V = VectorElement("U", cell, None)
        W = TensorElement("U", cell, None)

        u = Coefficient(U)
        v = Coefficient(V)
        w = Coefficient(W)
        du = Argument(U, 0)
        dv = Argument(V, 1)
        dw = Argument(W, 2)

        class ObjectCollection(object):
            pass
        self.shared_objects = ObjectCollection()
        for key, value in list(locals().items()):
            setattr(self.shared_objects, key, value)

        self.literals = list(map(as_ufl, [0, 1, 3.14, I, eps]))
        self.geometry = [x, n, c, h, f]
        self.functions = [u, du, v, dv, w, dw]

        self.terminals = []
        self.terminals += self.literals
        self.terminals += self.geometry
        self.terminals += self.functions

        self.algebra = ([
            u*2, v*2, w*2,
            u+2*u, v+2*v, w+2*w,
            2/u, u/2, v/2, w/2,
            u**3, 3**u,
        ])
        self.mathfunctions = ([
            abs(u), sqrt(u), exp(u), ln(u),
            cos(u), sin(u), tan(u), acos(u), asin(u), atan(u),
            erf(u), bessel_I(1, u), bessel_J(1, u), bessel_K(1, u), bessel_Y(1, u),
        ])
        self.variables = ([
            variable(u), variable(v), variable(w),
            variable(w*u), 3*variable(w*u),
        ])

        if d == 1:
            w2 = as_matrix(((u**2,),))
        if d == 2:
            w2 = as_matrix(((u**2, u**3),
                            (u**4, u**5)))
        if d == 3:
            w2 = as_matrix(((u**2, u**3, u**4),
                            (u**4, u**5, u**6),
                            (u**6, u**7, u**8)))

        # Indexed,  ListTensor, ComponentTensor, IndexSum
        i, j, k, l = indices(4)
        self.indexing = ([
                v[0], w[d-1, 0], v[i], w[i, j],
                v[:], w[0,:], w[:, 0],
                v[...], w[0, ...], w[..., 0],
                v[i]*v[j], w[i, 0]*v[j], w[d-1, j]*v[i],
                v[i]*v[i], w[i, 0]*w[0, i], v[i]*w[0, i],
                v[j]*w[d-1, j], w[i, i], w[i, j]*w[j, i],
                as_tensor(v[i]*w[k, 0], (k, i)),
                as_tensor(v[i]*w[k, 0], (k, i))[:, l],
                as_tensor(w[i, j]*w[k, l], (k, j, l, i)),
                as_tensor(w[i, j]*w[k, l], (k, j, l, i))[0, 0, 0, 0],
                as_vector((u, 2, 3)),
                as_matrix(((u**2, u**3), (u**4, u**5))),
                as_vector((u, 2, 3))[i],
                w2[i, j]*w[i, j],
        ])
        self.conditionals = ([
            conditional(le(u, 1.0), 1, 0),
            conditional(eq(3.0, u), 1, 0),
            conditional(ne(sin(u), cos(u)), 1, 0),
            conditional(lt(sin(u), cos(u)), 1, 0),
            conditional(ge(sin(u), cos(u)), 1, 0),
            conditional(gt(sin(u), cos(u)), 1, 0),
            conditional(And(lt(u, 3), gt(u, 1)), 1, 0),
            conditional(Or(lt(u, 3), gt(u, 1)), 1, 0),
            conditional(Not(ge(u, 0.0)), 1, 0),
            conditional(le(u, 0.0), 1, 2),
            conditional(Not(ge(u, 0.0)), 1, 2),
            conditional(And(Not(ge(u, 0.0)), lt(u, 1.0)), 1, 2),
            conditional(le(u, 0.0), u**3, ln(u)),
        ])
        self.restrictions = [u('+'), u('-'), v('+'), v('-'), w('+'), w('-')]
        if d > 1:
            i, j = indices(2)
            self.restrictions += ([
                v('+')[i]*v('+')[i],
                v[i]('+')*v[i]('+'),
                (v[i]*v[i])('+'),
                (v[i]*v[j])('+')*w[i, j]('+'),
            ])

        self.noncompounds = []
        self.noncompounds += self.algebra
        self.noncompounds += self.mathfunctions
        self.noncompounds += self.variables
        self.noncompounds += self.indexing
        self.noncompounds += self.conditionals
        self.noncompounds += self.restrictions

        if d == 1:
            self.tensorproducts = []
        else:
            self.tensorproducts = ([
                dot(v, v),
                dot(v, w),
                dot(w, w),
                inner(v, v),
                inner(w, w),
                outer(v, v),
                outer(w, v),
                outer(v, w),
                outer(w, w),
            ])

        if d == 1:
            self.tensoralgebra = []
        else:
            self.tensoralgebra = ([
                w.T, sym(w), skew(w), dev(w),
                det(w), tr(w), cofac(w), inv(w),
            ])

        if d != 3:
            self.crossproducts = []
        else:
            self.crossproducts = ([
                cross(v, v),
                cross(v, 2*v),
                cross(v, w[0,:]),
                cross(v, w[:, 1]),
                cross(w[:, 0], v),
            ])

        self.compounds = []
        self.compounds += self.tensorproducts
        self.compounds += self.tensoralgebra
        self.compounds += self.crossproducts

        self.all_expressions = []
        self.all_expressions += self.terminals
        self.all_expressions += self.noncompounds
        self.all_expressions += self.compounds


@pytest.fixture(params=(1,2,3))
def d_expr(request):
    d = request.param
    cell = {1: interval, 2: triangle, 3: tetrahedron}[d]
    expr = ExpressionCollection(cell)
    return d, expr


def ad_algorithm(expr):
    #alt = 1
    #alt = 4
    #alt = 6
    alt = 0
    if alt == 0:
        return expand_derivatives(expr)
    elif alt == 1:
        return expand_derivatives(expr,
            apply_expand_compounds_before=True,
            apply_expand_compounds_after=False,
            use_alternative_wrapper_algorithm=True)
    elif alt == 2:
        return expand_derivatives(expr,
            apply_expand_compounds_before=False,
            apply_expand_compounds_after=True,
            use_alternative_wrapper_algorithm=False)
    elif alt == 3:
        return expand_derivatives(expr,
            apply_expand_compounds_before=False,
            apply_expand_compounds_after=False,
            use_alternative_wrapper_algorithm=False)
    elif alt == 4:
        return expand_derivatives(expr,
            apply_expand_compounds_before=False,
            apply_expand_compounds_after=False,
            use_alternative_wrapper_algorithm=True)
    elif alt == 5:
        return expand_derivatives(expr,
            apply_expand_compounds_before=False,
            apply_expand_compounds_after=False,
            use_alternative_wrapper_algorithm=False)


def _test_no_derivatives_no_change(self, collection):
    for expr in collection:
        before = expr
        after = ad_algorithm(before)
        #print '\n', str(before), '\n', str(after), '\n'
        self.assertEqualTotalShape(before, after)
        assert before == after


def _test_no_derivatives_but_still_changed(self, collection):
    # Planning to fix these:
    for expr in collection:
        before = expr
        after = ad_algorithm(before)
        #print '\n', str(before), '\n', str(after), '\n'
        self.assertEqualTotalShape(before, after)
        #assert before == after # Without expand_compounds
        self.assertNotEqual(before, after) # With expand_compounds


def test_only_terminals_no_change(self, d_expr):
    d, ex = d_expr
    _test_no_derivatives_no_change(self, ex.terminals)


def test_no_derivatives_no_change(self, d_expr):
    d, ex = d_expr
    _test_no_derivatives_no_change(self, ex.noncompounds)


def xtest_compounds_no_derivatives_no_change(self, d_expr): # This test fails with expand_compounds enabled
    d, ex = d_expr
    _test_no_derivatives_no_change(self, ex.compounds)


def test_zero_derivatives_of_terminals_produce_the_right_types_and_shapes(self, d_expr):
    d, ex = d_expr
    _test_zero_derivatives_of_terminals_produce_the_right_types_and_shapes(self, ex)


def _test_zero_derivatives_of_terminals_produce_the_right_types_and_shapes(self, collection):
    c = Constant(collection.shared_objects.cell)

    u = Coefficient(collection.shared_objects.U)
    v = Coefficient(collection.shared_objects.V)
    w = Coefficient(collection.shared_objects.W)

    for t in collection.terminals:
        for var in (u, v, w):
            before = derivative(t, var) # This will often get preliminary simplified to zero
            after = ad_algorithm(before)
            expected = 0*t
            #print '\n', str(expected), '\n', str(after), '\n', str(before), '\n'
            assert after == expected

            before = derivative(c*t, var) # This will usually not get simplified to zero
            after = ad_algorithm(before)
            expected = 0*t
            #print '\n', str(expected), '\n', str(after), '\n', str(before), '\n'
            assert after == expected


def test_zero_diffs_of_terminals_produce_the_right_types_and_shapes(self, d_expr):
    d, ex = d_expr
    _test_zero_diffs_of_terminals_produce_the_right_types_and_shapes(self, ex)


def _test_zero_diffs_of_terminals_produce_the_right_types_and_shapes(self, collection):
    c = Constant(collection.shared_objects.cell)

    u = Coefficient(collection.shared_objects.U)
    v = Coefficient(collection.shared_objects.V)
    w = Coefficient(collection.shared_objects.W)

    vu = variable(u)
    vv = variable(v)
    vw = variable(w)
    for t in collection.terminals:
        for var in (vu, vv, vw):
            before = diff(t, var) # This will often get preliminary simplified to zero
            after = ad_algorithm(before)
            expected = 0*outer(t, var)
            #print '\n', str(expected), '\n', str(after), '\n', str(before), '\n'
            assert after == expected

            before = diff(c*t, var) # This will usually not get simplified to zero
            after = ad_algorithm(before)
            expected = 0*outer(t, var)
            #print '\n', str(expected), '\n', str(after), '\n', str(before), '\n'
            assert after == expected


def test_zero_derivatives_of_noncompounds_produce_the_right_types_and_shapes(self, d_expr):
    d, ex = d_expr
    _test_zero_derivatives_of_noncompounds_produce_the_right_types_and_shapes(self, ex)


def _test_zero_derivatives_of_noncompounds_produce_the_right_types_and_shapes(self, collection):
    debug = 0

    u = Coefficient(collection.shared_objects.U)
    v = Coefficient(collection.shared_objects.V)
    w = Coefficient(collection.shared_objects.W)

    #for t in chain(collection.noncompounds, collection.compounds):
    #debug = True
    for t in collection.noncompounds:
        for var in (u, v, w):
            if debug:
                print('\n', 'shapes:   ', t.ufl_shape, var.ufl_shape, '\n')
            if debug:
                print('\n', 't:        ', str(t), '\n')
            if debug:
                print('\n', 't ind:    ', str(t.ufl_free_indices), '\n')
            if debug:
                print('\n', 'var:      ', str(var), '\n')
            before = derivative(t, var)
            if debug:
                print('\n', 'before:   ', str(before), '\n')
            after = ad_algorithm(before)
            if debug:
                print('\n', 'after:    ', str(after), '\n')
            expected = 0*t
            if debug:
                print('\n', 'expected: ', str(expected), '\n')
            assert after == expected


def test_zero_diffs_of_noncompounds_produce_the_right_types_and_shapes(self, d_expr):
    d, ex = d_expr
    _test_zero_diffs_of_noncompounds_produce_the_right_types_and_shapes(self, ex)


def _test_zero_diffs_of_noncompounds_produce_the_right_types_and_shapes(self, collection):
    debug = 0

    u = Coefficient(collection.shared_objects.U)
    v = Coefficient(collection.shared_objects.V)
    w = Coefficient(collection.shared_objects.W)

    vu = variable(u)
    vv = variable(v)
    vw = variable(w)

    #for t in chain(collection.noncompounds, collection.compounds):
    for t in collection.noncompounds:
        for var in (vu, vv, vw):
            before = diff(t, var)
            if debug:
                print('\n', 'before:   ', str(before), '\n')
            after = ad_algorithm(before)
            if debug:
                print('\n', 'after:    ', str(after), '\n')
            expected = 0*outer(t, var)
            if debug:
                print('\n', 'expected: ', str(expected), '\n')
            #print '\n', str(expected), '\n', str(after), '\n', str(before), '\n'
            assert after == expected


def test_nonzero_derivatives_of_noncompounds_produce_the_right_types_and_shapes(self, d_expr):
    d, ex = d_expr
    _test_nonzero_derivatives_of_noncompounds_produce_the_right_types_and_shapes(self, ex)


def _test_nonzero_derivatives_of_noncompounds_produce_the_right_types_and_shapes(self, collection):
    debug = 0

    u = collection.shared_objects.u
    v = collection.shared_objects.v
    w = collection.shared_objects.w

    #for t in chain(collection.noncompounds, collection.compounds):
    for t in collection.noncompounds:
        for var in (u, v, w):
            # Include d/dx [z ? y: x] but not d/dx [x ? f: z]
            if isinstance(t, Conditional) and (var in unique_post_traversal(t.ufl_operands[0])):
                if debug:
                    print(("Depends on %s :: %s" % (str(var), str(t))))
                continue

            if debug:
                print(('\n', '...:   ', t.ufl_shape, var.ufl_shape, '\n'))
            before = derivative(t, var)
            if debug:
                print(('\n', 'before:   ', str(before), '\n'))
            after = ad_algorithm(before)
            if debug:
                print(('\n', 'after:    ', str(after), '\n'))
            expected_shape = 0*t
            if debug:
                print(('\n', 'expected_shape: ', str(expected_shape), '\n'))
            #print '\n', str(expected_shape), '\n', str(after), '\n', str(before), '\n'

            if var in unique_post_traversal(t):
                self.assertEqualTotalShape(after, expected_shape)
                self.assertNotEqual(after, expected_shape)
            else:
                assert after == expected_shape


def test_nonzero_diffs_of_noncompounds_produce_the_right_types_and_shapes(self, d_expr):
    d, ex = d_expr
    _test_nonzero_diffs_of_noncompounds_produce_the_right_types_and_shapes(self, ex)


def _test_nonzero_diffs_of_noncompounds_produce_the_right_types_and_shapes(self, collection):
    debug = 0
    u = collection.shared_objects.u
    v = collection.shared_objects.v
    w = collection.shared_objects.w

    vu = variable(u)
    vv = variable(v)
    vw = variable(w)

    #for t in chain(collection.noncompounds, collection.compounds):
    for t in collection.noncompounds:
        t = replace(t, {u:vu, v:vv, w:vw})
        for var in (vu, vv, vw):
            # Include d/dx [z ? y: x] but not d/dx [x ? f: z]
            if isinstance(t, Conditional) and (var in unique_post_traversal(t.ufl_operands[0])):
                if debug:
                    print(("Depends on %s :: %s" % (str(var), str(t))))
                continue

            before = diff(t, var)
            if debug:
                print(('\n', 'before:   ', str(before), '\n'))
            after = ad_algorithm(before)
            if debug:
                print(('\n', 'after:    ', str(after), '\n'))
            expected_shape = 0*outer(t, var) # expected shape, not necessarily value
            if debug:
                print(('\n', 'expected_shape: ', str(expected_shape), '\n'))
            #print '\n', str(expected_shape), '\n', str(after), '\n', str(before), '\n'

            if var in unique_post_traversal(t):
                self.assertEqualTotalShape(after, expected_shape)
                self.assertNotEqual(after, expected_shape)
            else:
                assert after == expected_shape


def test_grad_coeff(self, d_expr):
    d, collection = d_expr

    u = collection.shared_objects.u
    v = collection.shared_objects.v
    w = collection.shared_objects.w
    for f in (u, v, w):
        before = grad(f)
        after = ad_algorithm(before)

        if before.ufl_shape != after.ufl_shape:
            print(('\n', 'shapes:', before.ufl_shape, after.ufl_shape))
            print(('\n', str(before), '\n', str(after), '\n'))

        self.assertEqualTotalShape(before, after)
        if f is u: # Differing by being wrapped in indexing types
            assert before == after

        before = grad(grad(f))
        after = ad_algorithm(before)
        self.assertEqualTotalShape(before, after)
        #assert before == after # Differing by being wrapped in indexing types

        before = grad(grad(grad(f)))
        after = ad_algorithm(before)
        self.assertEqualTotalShape(before, after)
        #assert before == after # Differing by being wrapped in indexing types


def test_derivative_grad_coeff(self, d_expr):
    d, collection = d_expr

    u = collection.shared_objects.u
    v = collection.shared_objects.v
    w = collection.shared_objects.w
    for f in (u, v, w):
        before = derivative(grad(f), f)
        after = ad_algorithm(before)
        self.assertEqualTotalShape(before, after)
        #assert after == expected

        before = derivative(grad(grad(f)), f)
        after = ad_algorithm(before)
        self.assertEqualTotalShape(before, after)
        #assert after == expected

        before = derivative(grad(grad(grad(f))), f)
        after = ad_algorithm(before)
        self.assertEqualTotalShape(before, after)
        #assert after == expected
        if 0:
            print()
            print(('B', f, "::", before))
            print(('A', f, "::", after))


def xtest_derivative_grad_coeff_with_variation_components(self, d_expr):
    d, collection = d_expr

    v = collection.shared_objects.v
    w = collection.shared_objects.w
    dv = collection.shared_objects.dv
    dw = collection.shared_objects.dw
    for g, dg in ((v, dv), (w, dw)):
        # Pick a single component
        ii = (0,)*(len(g.ufl_shape))
        f = g[ii]
        df = dg[ii]

        before = derivative(grad(g), f, df)
        after = ad_algorithm(before)
        self.assertEqualTotalShape(before, after)
        #assert after == expected

        before = derivative(grad(grad(g)), f, df)
        after = ad_algorithm(before)
        self.assertEqualTotalShape(before, after)
        #assert after == expected

        before = derivative(grad(grad(grad(g))), f, df)
        after = ad_algorithm(before)
        self.assertEqualTotalShape(before, after)
        #assert after == expected
        if 0:
            print()
            print(('B', f, "::", before))
            print(('A', f, "::", after))
