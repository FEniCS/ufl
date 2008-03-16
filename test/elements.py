#!/usr/bin/env python

import unittest

from ufl import *

# disable log output
import logging
logging.basicConfig(level=logging.CRITICAL)


# TODO: cover all valid element definitions

all_polygons = ("interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron")

class ElementsTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_scalar_cg(self):
        for dom in all_polygons:
            for p in range(1,10):
                element = FiniteElement("Lagrange", dom, p)
                self.assertTrue(element.value_rank() == 0)
                element = FiniteElement("CG",       dom, p)
                self.assertTrue(element.value_rank() == 0)

    def test_scalar_dg(self):
        for dom in all_polygons:
            for p in range(1,10):
                element = FiniteElement("Discontinuous Lagrange", dom, p)
                self.assertTrue(element.value_rank() == 0)
                element = FiniteElement("DG",                     dom, p)
                self.assertTrue(element.value_rank() == 0)

    def test_vector_cg(self):
        for dom in all_polygons:
            for p in range(1,10):
                element = VectorElement("Lagrange", dom, p)
                self.assertTrue(element.value_rank() == 1)
                element = VectorElement("CG",       dom, p)
                self.assertTrue(element.value_rank() == 1)

    def test_vector_dg(self):
        for dom in all_polygons:
            for p in range(1,10):
                element = VectorElement("Discontinuous Lagrange", dom, p)
                self.assertTrue(element.value_rank() == 1)
                element = VectorElement("DG",                     dom, p)
                self.assertTrue(element.value_rank() == 1)

    def test_tensor_cg(self):
        for dom in all_polygons:
            for p in range(1,10):
                element = TensorElement("Lagrange", dom, p)
                self.assertTrue(element.value_rank() == 2)
                element = TensorElement("CG",       dom, p)
                self.assertTrue(element.value_rank() == 2)

    def test_tensor_dg(self):
        for dom in all_polygons:
            for p in range(1,10):
                element = TensorElement("Discontinuous Lagrange", dom, p)
                self.assertTrue(element.value_rank() == 2)
                element = TensorElement("DG",                     dom, p)
                self.assertTrue(element.value_rank() == 2)

    def test_bdm(self):
        element = FiniteElement("BDM", "triangle", 1)
        self.assertTrue(element.value_rank() == 1)
        element = FiniteElement("BDM", "tetrahedron", 1)
        self.assertTrue(element.value_rank() == 1)

    def test_vector_bdm(self):
        element = VectorElement("BDM", "triangle", 1)
        self.assertTrue(element.value_rank() == 2)
        element = VectorElement("BDM", "tetrahedron", 1)
        self.assertTrue(element.value_rank() == 2)

    def test_mixed(self):
        velement = VectorElement("CG", "triangle", 2)
        pelement = FiniteElement("CG", "triangle", 1)
        TH1 = MixedElement(velement, pelement)
        TH2 = velement + pelement
        assert repr(TH1) == repr(TH2)
        try:
            TH1.value_rank()
            self.fail()
        except:
            pass


suite1 = unittest.makeSuite(ElementsTestCase)

allsuites = unittest.TestSuite((suite1, ))

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=0).run(allsuites)
