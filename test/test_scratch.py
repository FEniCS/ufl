#!/usr/bin/env py.test
# -*- coding: utf-8 -*-
"""
This is a template file you can copy when making a new test case.
Begin by copying this file to a filename matching test_*.py.
The tests in the file will then automatically be run by ./test.py.
Next look at the TODO markers below for places to edit.
"""

import pytest
from six.moves import zip

# This imports everything external code will see from ufl
from ufl import *
from ufl.log import error, warning
from ufl.assertions import ufl_assert
from ufl.tensors import as_scalar, unit_indexed_tensor, unwrap_list_tensor

# TODO: Import only what you need from classes and algorithms:
from ufl.classes import Grad, FormArgument, Zero, Indexed, FixedIndex, ListTensor

class MockForwardAD:
    def __init__(self):
        self._w = ()
        self._v = ()
        class Obj:
            def __init__(self):
                self._data = {}
        self._cd = Obj()

    def grad(self, g):
        # If we hit this type, it has already been propagated
        # to a coefficient (or grad of a coefficient), # FIXME: Assert this!
        # so we need to take the gradient of the variation or return zero.
        # Complications occur when dealing with derivatives w.r.t. single components...

        # Figure out how many gradients are around the inner terminal
        ngrads = 0
        o = g
        while isinstance(o, Grad):
            o, = o.ufl_operands
            ngrads += 1
        if not isinstance(o, FormArgument):
            error("Expecting gradient of a FormArgument, not %r" % (o,))

        def apply_grads(f):
            if not isinstance(f, FormArgument):
                print((','*60))
                print(f)
                print(o)
                print(g)
                print((','*60))
                error("What?")
            for i in range(ngrads):
                f = Grad(f)
            return f

        # Find o among all w without any indexing, which makes this easy
        for (w, v) in zip(self._w, self._v):
            if o == w and isinstance(v, FormArgument):
                # Case: d/dt [w + t v]
                return (g, apply_grads(v))

        # If o is not among coefficient derivatives, return do/dw=0
        gprimesum = Zero(g.ufl_shape)

        def analyse_variation_argument(v):
            # Analyse variation argument
            if isinstance(v, FormArgument):
                # Case: d/dt [w[...] + t v]
                vval, vcomp = v, ()
            elif isinstance(v, Indexed):
                # Case: d/dt [w + t v[...]]
                # Case: d/dt [w[...] + t v[...]]
                vval, vcomp = v.ufl_operands
                vcomp = tuple(vcomp)
            else:
                error("Expecting argument or component of argument.")
            ufl_assert(all(isinstance(k, FixedIndex) for k in vcomp),
                       "Expecting only fixed indices in variation.")
            return vval, vcomp

        def compute_gprimeterm(ngrads, vval, vcomp, wshape, wcomp):
            # Apply gradients directly to argument vval,
            # and get the right indexed scalar component(s)
            kk = indices(ngrads)
            Dvkk = apply_grads(vval)[vcomp+kk]
            # Place scalar component(s) Dvkk into the right tensor positions
            if wshape:
                Ejj, jj = unit_indexed_tensor(wshape, wcomp)
            else:
                Ejj, jj = 1, ()
            gprimeterm = as_tensor(Ejj*Dvkk, jj+kk)
            return gprimeterm

        # Accumulate contributions from variations in different components
        for (w, v) in zip(self._w, self._v):

            # Analyse differentiation variable coefficient
            if isinstance(w, FormArgument):
                if not w == o: continue
                wshape = w.ufl_shape

                if isinstance(v, FormArgument):
                    # Case: d/dt [w + t v]
                    return (g, apply_grads(v))

                elif isinstance(v, ListTensor):
                    # Case: d/dt [w + t <...,v,...>]
                    for wcomp, vsub in unwrap_list_tensor(v):
                        if not isinstance(vsub, Zero):
                            vval, vcomp = analyse_variation_argument(vsub)
                            gprimesum = gprimesum + compute_gprimeterm(ngrads, vval, vcomp, wshape, wcomp)

                else:
                    ufl_assert(wshape == (), "Expecting scalar coefficient in this branch.")
                    # Case: d/dt [w + t v[...]]
                    wval, wcomp = w, ()

                    vval, vcomp = analyse_variation_argument(v)
                    gprimesum = gprimesum + compute_gprimeterm(ngrads, vval, vcomp, wshape, wcomp)

            elif isinstance(w, Indexed): # This path is tested in unit tests, but not actually used?
                # Case: d/dt [w[...] + t v[...]]
                # Case: d/dt [w[...] + t v]
                wval, wcomp = w.ufl_operands
                if not wval == o: continue
                assert isinstance(wval, FormArgument)
                ufl_assert(all(isinstance(k, FixedIndex) for k in wcomp),
                           "Expecting only fixed indices in differentiation variable.")
                wshape = wval.ufl_shape

                vval, vcomp = analyse_variation_argument(v)
                gprimesum = gprimesum + compute_gprimeterm(ngrads, vval, vcomp, wshape, wcomp)

            else:
                error("Expecting coefficient or component of coefficient.")

        # FIXME: Handle other coefficient derivatives: oprimes = self._cd._data.get(o)

        if 0:
            oprimes = self._cd._data.get(o)
            if oprimes is None:
                if self._cd._data:
                    # TODO: Make it possible to silence this message in particular?
                    #       It may be good to have for debugging...
                    warning("Assuming d{%s}/d{%s} = 0." % (o, self._w))
            else:
                # Make sure we have a tuple to match the self._v tuple
                if not isinstance(oprimes, tuple):
                    oprimes = (oprimes,)
                    ufl_assert(len(oprimes) == len(self._v), "Got a tuple of arguments, "+\
                                   "expecting a matching tuple of coefficient derivatives.")

                # Compute dg/dw_j = dg/dw_h : v.
                # Since we may actually have a tuple of oprimes and vs in a
                # 'mixed' space, sum over them all to get the complete inner
                # product. Using indices to define a non-compound inner product.
                for (oprime, v) in zip(oprimes, self._v):
                    error("FIXME: Figure out how to do this with ngrads")
                    so, oi = as_scalar(oprime)
                    rv = len(v.ufl_shape)
                    oi1 = oi[:-rv]
                    oi2 = oi[-rv:]
                    prod = so*v[oi2]
                    if oi1:
                        gprimesum += as_tensor(prod, oi1)
                    else:
                        gprimesum += prod

        return (g, gprimesum)


def test_unit_tensor(self):
    E2_1, ii = unit_indexed_tensor((2,), (1,))
    E3_1, ii = unit_indexed_tensor((3,), (1,))
    E22_10, ii = unit_indexed_tensor((2, 2), (1, 0))
    # TODO: Evaluate and assert values

def test_unwrap_list_tensor(self):
    lt = as_tensor((1, 2))
    expected = [((0,), 1),
                ((1,), 2),]
    comp = unwrap_list_tensor(lt)
    assert comp == expected

    lt = as_tensor(((1, 2), (3, 4)))
    expected = [((0, 0), 1),
                ((0, 1), 2),
                ((1, 0), 3),
                ((1, 1), 4),]
    comp = unwrap_list_tensor(lt)
    assert comp == expected

    lt = as_tensor((((1, 2), (3, 4)),
                    ((11, 12), (13, 14))))
    expected = [((0, 0, 0), 1),
                ((0, 0, 1), 2),
                ((0, 1, 0), 3),
                ((0, 1, 1), 4),
                ((1, 0, 0), 11),
                ((1, 0, 1), 12),
                ((1, 1, 0), 13),
                ((1, 1, 1), 14),]
    comp = unwrap_list_tensor(lt)
    assert comp == expected

def test__forward_coefficient_ad__grad_of_scalar_coefficient(self):
    U = FiniteElement("CG", triangle, 1)
    u = Coefficient(U)
    du = TestFunction(U)

    mad = MockForwardAD()
    mad._w = (u,)
    mad._v = (du,)

    # Simple grad(coefficient) -> grad(variation)
    f = grad(u)
    df = grad(du)
    g, dg = mad.grad(f)
    assert g == f
    assert dg == df

    # Simple grad(grad(coefficient)) -> grad(grad(variation))
    f = grad(grad(u))
    df = grad(grad(du))
    g, dg = mad.grad(f)
    assert g == f
    assert dg == df

def test__forward_coefficient_ad__grad_of_vector_coefficient(self):
    V = VectorElement("CG", triangle, 1)
    v = Coefficient(V)
    dv = TestFunction(V)

    mad = MockForwardAD()
    mad._w = (v,)
    mad._v = (dv,)

    # Simple grad(coefficient) -> grad(variation)
    f = grad(v)
    df = grad(dv)
    g, dg = mad.grad(f)
    assert g == f
    assert dg == df

    # Simple grad(grad(coefficient)) -> grad(grad(variation))
    f = grad(grad(v))
    df = grad(grad(dv))
    g, dg = mad.grad(f)
    assert g == f
    assert dg == df

def test__forward_coefficient_ad__grad_of_vector_coefficient__with_component_variation(self):
    V = VectorElement("CG", triangle, 1)
    v = Coefficient(V)
    dv = TestFunction(V)

    mad = MockForwardAD()

    # Component of variation:
    # grad(grad(c))[0,...] -> grad(grad(dc))[1,...]
    mad._w = (v[0],)
    mad._v = (dv[1],)
    f = grad(v)
    df = grad(as_vector((dv[1], 0))) # Mathematically this would be the natural result
    j, k = indices(2)
    df = as_tensor(Identity(2)[0, j]*grad(dv)[1, k], (j, k)) # Actual representation should have grad right next to dv
    g, dg = mad.grad(f)
    if 0:
        print(('\nf    ', f))
        print(('df   ', df))
        print(('g    ', g))
        print(('dg   ', dg))
    assert f.ufl_shape == df.ufl_shape
    assert g.ufl_shape == f.ufl_shape
    assert dg.ufl_shape == df.ufl_shape
    assert g == f
    self.assertEqual((inner(dg, dg)*dx).signature(),
                     (inner(df, df)*dx).signature())
    #assert dg == df # Expected to fail because of different index numbering

    # Multiple components of variation:
    # grad(grad(c))[0,1,:,:] -> grad(grad(dc))[1,0,:,:]
    mad._w = (v[0], v[1])
    mad._v = (dv[1], dv[0])
    f = grad(v)
    # Mathematically this would be the natural result:
    df = grad(as_vector((dv[1], dv[0])))
    # Actual representation should have grad right next to dv:
    j0, k0 = indices(2)
    j1, k1 = indices(2) # Using j0,k0 for both terms gives different signature
    df = (as_tensor(Identity(2)[0, j0]*grad(dv)[1, k0], (j0, k0))
        + as_tensor(Identity(2)[1, j1]*grad(dv)[0, k1], (j1, k1)))
    g, dg = mad.grad(f)
    print(('\nf    ', f))
    print(('df   ', df))
    print(('g    ', g))
    print(('dg   ', dg))
    assert f.ufl_shape == df.ufl_shape
    assert g.ufl_shape == f.ufl_shape
    assert dg.ufl_shape == df.ufl_shape
    assert g == f
    self.assertEqual((inner(dg, dg)*dx).signature(),
                     (inner(df, df)*dx).signature())
    #assert dg == df # Expected to fail because of different index numbering

def test__forward_coefficient_ad__grad_of_vector_coefficient__with_component_variation_in_list(self):
    V = VectorElement("CG", triangle, 1)
    v = Coefficient(V)
    dv = TestFunction(V)

    mad = MockForwardAD()

    # Component of variation:
    # grad(grad(c))[0,...] -> grad(grad(dc))[1,...]
    mad._w = (v,)
    mad._v = (as_vector((dv[1], 0)),)
    f = grad(v)
    df = grad(as_vector((dv[1], 0))) # Mathematically this would be the natural result
    j, k = indices(2)
    df = as_tensor(Identity(2)[0, j]*grad(dv)[1, k], (j, k)) # Actual representation should have grad right next to dv
    g, dg = mad.grad(f)
    if 0:
        print(('\nf    ', f))
        print(('df   ', df))
        print(('g    ', g))
        print(('dg   ', dg))
    assert f.ufl_shape == df.ufl_shape
    assert g.ufl_shape == f.ufl_shape
    assert dg.ufl_shape == df.ufl_shape
    assert g == f
    self.assertEqual((inner(dg, dg)*dx).signature(),
                     (inner(df, df)*dx).signature())
    #assert dg == df # Expected to fail because of different index numbering

    # Multiple components of variation:
    # grad(grad(c))[0,1,:,:] -> grad(grad(dc))[1,0,:,:]
    mad._w = (v, )
    mad._v = (as_vector((dv[1], dv[0])),)
    f = grad(v)
    # Mathematically this would be the natural result:
    df = grad(as_vector((dv[1], dv[0])))
    # Actual representation should have grad right next to dv:
    j0, k0 = indices(2)
    j1, k1 = indices(2) # Using j0,k0 for both terms gives different signature
    df = (as_tensor(Identity(2)[0, j0]*grad(dv)[1, k0], (j0, k0))
        + as_tensor(Identity(2)[1, j1]*grad(dv)[0, k1], (j1, k1)))
    g, dg = mad.grad(f)
    print(('\nf    ', f))
    print(('df   ', df))
    print(('g    ', g))
    print(('dg   ', dg))
    assert f.ufl_shape == df.ufl_shape
    assert g.ufl_shape == f.ufl_shape
    assert dg.ufl_shape == df.ufl_shape
    assert g == f
    self.assertEqual((inner(dg, dg)*dx).signature(),
                     (inner(df, df)*dx).signature())
    #assert dg == df # Expected to fail because of different index numbering


def test__forward_coefficient_ad__grad_of_tensor_coefficient(self):
    W = TensorElement("CG", triangle, 1)
    w = Coefficient(W)
    dw = TestFunction(W)

    mad = MockForwardAD()
    mad._w = (w,)
    mad._v = (dw,)

    # Simple grad(coefficient) -> grad(variation)
    f = grad(w)
    df = grad(dw)
    g, dg = mad.grad(f)
    assert g == f
    assert dg == df

    # Simple grad(grad(coefficient)) -> grad(grad(variation))
    f = grad(grad(w))
    df = grad(grad(dw))
    g, dg = mad.grad(f)
    assert g == f
    assert dg == df

def test__forward_coefficient_ad__grad_of_tensor_coefficient__with_component_variation(self):
    W = TensorElement("CG", triangle, 1)
    w = Coefficient(W)
    dw = TestFunction(W)

    mad = MockForwardAD()

    # Component of variation:
    # grad(grad(c))[0,...] -> grad(grad(dc))[1,...]
    wc = (1, 0)
    dwc = (0, 1)
    mad._w = (w[wc],)
    mad._v = (dw[dwc],)
    f = grad(w)
    df = grad(as_matrix(((0, 0), (dw[dwc], 0)))) # Mathematically this is it.
    i, j, k = indices(3)
    E = outer(Identity(2)[wc[0], i], Identity(2)[wc[1], j])
    Ddw = grad(dw)[dwc + (k,)]
    df = as_tensor(E*Ddw, (i, j, k)) # Actual representation should have grad next to dv
    g, dg = mad.grad(f)
    if 0:
        print(('\nf    ', f))
        print(('df   ', df))
        print(('g    ', g))
        print(('dg   ', dg))
    assert f.ufl_shape == df.ufl_shape
    assert g.ufl_shape == f.ufl_shape
    assert dg.ufl_shape == df.ufl_shape
    assert g == f
    self.assertEqual((inner(dg, dg)*dx).signature(),
                     (inner(df, df)*dx).signature())
    #assert dg == df # Expected to fail because of different index numbering
