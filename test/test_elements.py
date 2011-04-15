#!/usr/bin/env python

# Last changed: 2009-12-08

from ufltestcase import UflTestCase, main

from ufl import *

from ufl.geometry import domain2dim

all_cells = (interval, triangle, tetrahedron, quadrilateral, hexahedron)

# TODO: cover all valid element definitions

class ElementsTestCase(UflTestCase):

    def test_scalar_galerkin(self):
        for cell in all_cells:
            for p in range(1,10):
                for family in ("Lagrange", "CG", "Discontinuous Lagrange", "DG"):
                    element = FiniteElement(family, cell, p)
                    self.assertEqual(element.value_shape(), ())

    def test_vector_galerkin(self):
        for cell in all_cells:
            dim = cell.d
            for p in range(1,10):
                for family in ("Lagrange", "CG", "Discontinuous Lagrange", "DG"):
                    element = VectorElement(family, cell, p)
                    self.assertEqual(element.value_shape(), (dim,))
                    for i in range(dim):
                        c = element.extract_component(i)
                        self.assertEqual(c[0], ())

    def test_tensor_galerkin(self):
        for cell in all_cells:
            dim = cell.d
            for p in range(1,10):
                for family in ("Lagrange", "CG", "Discontinuous Lagrange", "DG"):
                    element = TensorElement(family, cell, p)
                    self.assertEqual(element.value_shape(), (dim,dim))
                    for i in range(dim):
                        for j in range(dim):
                            c = element.extract_component((i,j))
                            self.assertEqual(c[0], ())

    def test_tensor_symmetry(self):
        for cell in all_cells:
            dim = cell.d
            for p in range(1,10):
                for s in (None, True, {(0,1): (1,0)}):
                    for family in ("Lagrange", "CG", "Discontinuous Lagrange", "DG"):
                        if isinstance(s, dict):
                            element = TensorElement(family, cell, p, shape=(dim,dim), symmetry=s)
                        else:
                            element = TensorElement(family, cell, p, symmetry=s)
                        self.assertEqual(element.value_shape(), (dim,dim))
                        for i in range(dim):
                            for j in range(dim):
                                c = element.extract_component((i,j))
                                self.assertEqual(c[0], ())

    def test_bdm(self):
        for cell in (triangle, tetrahedron):
            dim = cell.d
            element = FiniteElement("BDM", cell, 1)
            self.assertEqual(element.value_shape(), (dim,))

    def test_vector_bdm(self):
        for cell in (triangle, tetrahedron):
            dim = cell.d
            element = VectorElement("BDM", cell, 1)
            self.assertEqual(element.value_shape(), (dim,dim))

    def test_mixed(self):
        for cell in (triangle, tetrahedron):
            dim = cell.d
            velement = VectorElement("CG", cell, 2)
            pelement = FiniteElement("CG", cell, 1)
            TH1 = MixedElement(velement, pelement)
            TH2 = velement * pelement
            self.assertEqual(repr(TH1), repr(TH2))
            self.assertEqual(TH1.value_shape(), (dim+1,))
            self.assertEqual(TH2.value_shape(), (dim+1,))

if __name__ == "__main__":
    main()
