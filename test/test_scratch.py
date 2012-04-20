#!/usr/bin/env python

"""
This is a template file you can copy when making a new test case.
Begin by copying this file to a filename matching test_*.py.
The tests in the file will then automatically be run by ./test.py.
Next look at the TODO markers below for places to edit.
"""

# These are thin wrappers on top of unittest.TestCase and unittest.main
from ufltestcase import UflTestCase, main
from itertools import izip
# This imports everything external code will see from ufl
from ufl import *
from ufl.log import error, warning
from ufl.assertions import ufl_assert
from ufl.tensors import as_scalar, unit_list

# TODO: Import only what you need from classes and algorithms:
from ufl.classes import Grad, FormArgument, Zero, Indexed, FixedIndex
#from ufl.algorithms import ...

def unit_indexed_tensor(shape, component):
    r = len(shape)
    if r == 0:
        return 0, ()
    jj = indices(r)
    es = []
    for i in xrange(r):
        s = shape[i]
        c = component[i]
        j = jj[i]
        e = Identity(s)[c,j]
        es.append(e)
    E = es[0]
    for e in es[1:]:
        E = outer(E, e)
    return E, jj

class MockForwardAD:
    def __init__(self):
        self._w = ()
        self._v = ()
        class Obj:
            def __init__(self):
                self._data = {}
        self._cd = Obj()

    def grad(self, g): # FIXME: Fix implementation below, check error("FIXME...")
        # If we hit this type, it has already been propagated
        # to a coefficient (or grad of a coefficient), # FIXME: Assert this!
        # so we need to take the gradient of the variation or return zero.
        # Complications occur when dealing with derivatives w.r.t. single components...

        # Figure out how many gradients are around the inner terminal
        ngrads = 0
        o = g
        while isinstance(o, Grad):
            o, = o.operands()
            ngrads += 1
        if not isinstance(o, FormArgument):
            error("Expecting gradient of a FormArgument, not %r" % (o,))

        def apply_grads(f):
            if not isinstance(f, FormArgument):
                print ','*60
                print f
                print o
                print g
                print ','*60
                error("What?")
            for i in range(ngrads):
                f = Grad(f)
            return f

        # Find o among all w without any indexing, which makes this easy
        for (w, v) in izip(self._w, self._v):
            if o == w and isinstance(v, FormArgument):
                return (g, apply_grads(v))

        # FIXME: Apply gradients to everything below:

        # If o is not among coefficient derivatives, return do/dw=0
        gprimesum = Zero(g.shape())

        for (w, v) in izip(self._w, self._v):
            if w == o:
                if isinstance(v, FormArgument):
                    return (g, apply_grads(v))
                else:
                    wval = w
                    wcomp = None
            elif isinstance(w, Indexed):
                wval, wcomp = w.operands()
                if not wval == o:
                    continue
                assert isinstance(wval, FormArgument)
            else:
                FIXME

            ufl_assert(all(isinstance(k, FixedIndex) for k in wcomp),
                       "Expecting only fixed indices in differentiation variable.")

            wshape = wval.shape()
            if wcomp:
                Ejj, jj = unit_indexed_tensor(wshape, wcomp)
            else:
                FIXME
                Ejj, jj = 1, ()

            if isinstance(v, FormArgument):
                vval, vcomp = v, ()
            elif isinstance(v, Indexed):
                vval, vcomp = v.operands()
                vcomp = tuple(vcomp)
            else:
                FIXME

            ufl_assert(all(isinstance(k, FixedIndex) for k in vcomp),
                       "Expecting only fixed indices in variation.")

            kk = indices(ngrads)
            Dvkk = apply_grads(vval)[vcomp+kk]

            gprimeterm = as_tensor(Ejj*Dvkk, jj+kk)

            gprimesum = gprimesum + gprimeterm

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
                for (oprime, v) in izip(oprimes, self._v):
                    error("FIXME: Figure out how to do this with ngrads")
                    so, oi = as_scalar(oprime)
                    rv = len(v.shape())
                    oi1 = oi[:-rv]
                    oi2 = oi[-rv:]
                    prod = so*v[oi2]
                    if oi1:
                        gprimesum += as_tensor(prod, oi1)
                    else:
                        gprimesum += prod

        return (g, gprimesum)

class ScratchTestCase(UflTestCase):

    def setUp(self):
        super(ScratchTestCase, self).setUp()

    def tearDown(self):
        super(ScratchTestCase, self).tearDown()

    def test_something(self):
        self.assertTrue(42)

    def test__forward_coefficient_ad__grad_of_scalar_coefficient(self):
        U = FiniteElement("CG", cell2D, 1)
        u = Coefficient(U)
        du = TestFunction(U)

        mad = MockForwardAD()
        mad._w = (u,)
        mad._v = (du,)

        # Simple grad(coefficient) -> grad(variation)
        f = grad(u)
        df = grad(du)
        g, dg = mad.grad(f)
        self.assertEqual(g, f)
        self.assertEqual(dg, df)

        # Simple grad(grad(coefficient)) -> grad(grad(variation))
        f = grad(grad(u))
        df = grad(grad(du))
        g, dg = mad.grad(f)
        self.assertEqual(g, f)
        self.assertEqual(dg, df)

    def test_unit_tensor(self):
        print
        print unit_indexed_tensor((2,), (1,))
        print unit_indexed_tensor((3,), (1,))
        print unit_indexed_tensor((2,2), (1,0))

    def test__forward_coefficient_ad__grad_of_vector_coefficient(self):
        V = VectorElement("CG", cell2D, 1)
        v = Coefficient(V)
        dv = TestFunction(V)

        mad = MockForwardAD()
        mad._w = (v,)
        mad._v = (dv,)

        # Simple grad(coefficient) -> grad(variation)
        f = grad(v)
        df = grad(dv)
        g, dg = mad.grad(f)
        self.assertEqual(g, f)
        self.assertEqual(dg, df)

        # Simple grad(grad(coefficient)) -> grad(grad(variation))
        f = grad(grad(v))
        df = grad(grad(dv))
        g, dg = mad.grad(f)
        self.assertEqual(g, f)
        self.assertEqual(dg, df)

        # Component of variation:
        # grad(grad(c))[0,...] -> grad(grad(dc))[1,...]
        mad._w = (v[0],)
        mad._v = (dv[1],)
        f = grad(v)
        df = grad(as_vector((dv[1],0))) # Mathematically this is it.
        j,k = indices(2)
        df = as_tensor(Identity(2)[0,j]*grad(dv)[1,k], (j,k)) # Actual representation should have grad next to dv
        g, dg = mad.grad(f)
        print '\nf    ', f
        print 'df   ', df
        print 'g    ', g
        print 'dg   ', dg
        self.assertEqual(f.shape(), df.shape())
        self.assertEqual(g.shape(), f.shape())
        self.assertEqual(dg.shape(), df.shape())
        self.assertEqual(g, f)
        self.assertEqual((inner(dg,dg)*dx).signature(), (inner(df,df)*dx).signature())
        #self.assertEqual(dg, df) # Expected to fail because of different index numbering

    def test__forward_coefficient_ad__grad_of_tensor_coefficient(self):
        W = TensorElement("CG", cell2D, 1)
        w = Coefficient(W)
        dw = TestFunction(W)

        mad = MockForwardAD()
        mad._w = (w,)
        mad._v = (dw,)

        # Simple grad(coefficient) -> grad(variation)
        f = grad(w)
        df = grad(dw)
        g, dg = mad.grad(f)
        self.assertEqual(g, f)
        self.assertEqual(dg, df)

        # Simple grad(grad(coefficient)) -> grad(grad(variation))
        f = grad(grad(w))
        df = grad(grad(dw))
        g, dg = mad.grad(f)
        self.assertEqual(g, f)
        self.assertEqual(dg, df)

# Don't touch these lines, they allow you to run this file directly
if __name__ == "__main__":
    main()

