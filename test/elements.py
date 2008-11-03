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
                    self.assertTrue(element.value_shape() == ())

    def test_vector_galerkin(self):
        for dom in all_polygons:
            dim = domain2dim[dom]
            for p in range(1,10):
                for family in ("Lagrange", "CG", "Discontinuous Lagrange", "DG"):
                    element = VectorElement(family, dom, p)
                    self.assertTrue(element.value_shape() == (dim,))
                    for i in range(dim):
                        c = element.extract_component(i)
                        self.assertTrue(c[0] == ())

    def test_tensor_galerkin(self):
        for dom in all_polygons:
            dim = domain2dim[dom]
            for p in range(1,10):
                for family in ("Lagrange", "CG", "Discontinuous Lagrange", "DG"):
                    element = TensorElement(family, dom, p)
                    self.assertTrue(element.value_shape() == (dim,dim))
                    for i in range(dim):
                        for j in range(dim):
                            c = element.extract_component((i,j))
                            self.assertTrue(c[0] == ())

    def test_tensor_symmetry(self):
        for dom in all_polygons:
            dim = domain2dim[dom]
            for p in range(1,10):
                for s in (None, True, {(0,1): (1,0)}):
                    for family in ("Lagrange", "CG", "Discontinuous Lagrange", "DG"):
                        if isinstance(s, dict):
                            element = TensorElement(family, dom, p, shape=(dim,dim), symmetry=s)
                        else:
                            element = TensorElement(family, dom, p, symmetry=s)
                        self.assertTrue(element.value_shape() == (dim,dim))
                        for i in range(dim):
                            for j in range(dim):
                                c = element.extract_component((i,j))
                                self.assertTrue(c[0] == ())

    def test_bdm(self):
        for dom in ("triangle", "tetrahedron"):
            dim = domain2dim[dom]
            element = FiniteElement("BDM", dom, 1)
            self.assertTrue(element.value_shape() == (dim,))

    def test_vector_bdm(self):
        for dom in ("triangle", "tetrahedron"):
            dim = domain2dim[dom]
            element = VectorElement("BDM", dom, 1)
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


if __name__ == "__main__":
    unittest.main()
