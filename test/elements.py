#!/usr/bin/env python

import unittest

from ufl import *

# disable log output
import logging
logging.basicConfig(level=logging.CRITICAL)


# TODO: cover all valid element definitions

all_polygons = ("interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron")
domain2dim = {"interval": 1, "triangle": 2, "tetrahedron": 3, "quadrilateral": 2, "hexahedron": 3}

class ElementsTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_scalar_galerkin(self):
        for dom in all_polygons:
            for p in range(1,10):
                for family in ("Lagrange", "CG", "Discontinuous Lagrange", "DG"):
                    element = FiniteElement(family, dom, p)
                    self.assertTrue(element.value_rank() == 0)
                    self.assertTrue(element.value_shape() == ())

    def test_vector_galerkin(self):
        for dom in all_polygons:
            dim = domain2dim[dom]
            for p in range(1,10):
                for family in ("Lagrange", "CG", "Discontinuous Lagrange", "DG"):
                    element = VectorElement(family, dom, p)
                    self.assertTrue(element.value_rank() == 1)
                    self.assertTrue(element.value_shape() == (dim,))

    def test_tensor_galerkin(self):
        for dom in all_polygons:
            dim = domain2dim[dom]
            for p in range(1,10):
                for family in ("Lagrange", "CG", "Discontinuous Lagrange", "DG"):
                    element = TensorElement(family, dom, p)
                    self.assertTrue(element.value_rank() == 2)
                    self.assertTrue(element.value_shape() == (dim,dim))

    def test_bdm(self):
        for dom in ("triangle", "tetrahedron"):
            dim = domain2dim[dom]
            element = FiniteElement("BDM", dom, 1)
            self.assertTrue(element.value_rank() == 1)
            self.assertTrue(element.value_shape() == (dim,))

    def test_vector_bdm(self):
        for dom in ("triangle", "tetrahedron"):
            dim = domain2dim[dom]
            element = VectorElement("BDM", dom, 1)
            self.assertTrue(element.value_rank() == 2)
            self.assertTrue(element.value_shape() == (dim,dim))

    def test_mixed(self):
        for dom in ("triangle", "tetrahedron"):
            dim = domain2dim[dom]
            velement = VectorElement("CG", dom, 2)
            pelement = FiniteElement("CG", dom, 1)
            TH1 = MixedElement(velement, pelement)
            TH2 = velement + pelement
            self.assertTrue( repr(TH1) == repr(TH2) )
            self.assertTrue( TH1.value_shape() == (dim+1,) )
            self.assertTrue( TH2.value_shape() == (dim+1,) )


suite1 = unittest.makeSuite(ElementsTestCase)

allsuites = unittest.TestSuite((suite1, ))

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=0).run(allsuites)
