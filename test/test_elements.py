#!/usr/bin/env python

# Last changed: 2009-12-08

from ufltestcase import UflTestCase, main

from ufl import *

from ufl.geometry import cellname2dim

all_cells = (interval, triangle, tetrahedron, quadrilateral, hexahedron)

# TODO: cover all valid element definitions

class ElementsTestCase(UflTestCase):

    def test_scalar_galerkin(self):
        for cell in all_cells:
            for p in range(1,10):
                for family in ("Lagrange", "CG", "Discontinuous Lagrange", "DG"):
                    element = FiniteElement(family, cell, p)
                    self.assertEqual(element.value_shape(), ())
                    self.assertEqual(element, eval(repr(element)))

    def test_vector_galerkin(self):
        for cell in all_cells:
            dim = cell.d
            #shape = () if dim == 1 else (dim,)
            shape = (dim,)
            for p in range(1,10):
                for family in ("Lagrange", "CG", "Discontinuous Lagrange", "DG"):
                    element = VectorElement(family, cell, p)
                    self.assertEqual(element.value_shape(), shape)
                    self.assertEqual(element, eval(repr(element)))
                    for i in range(dim):
                        c = element.extract_component(i)
                        self.assertEqual(c[0], ())

    def test_tensor_galerkin(self):
        for cell in all_cells:
            dim = cell.d
            #shape = () if dim == 1 else (dim,dim)
            shape = (dim,dim)
            for p in range(1,10):
                for family in ("Lagrange", "CG", "Discontinuous Lagrange", "DG"):
                    element = TensorElement(family, cell, p)
                    self.assertEqual(element.value_shape(), shape)
                    self.assertEqual(element, eval(repr(element)))
                    for i in range(dim):
                        for j in range(dim):
                            c = element.extract_component((i,j))
                            self.assertEqual(c[0], ())

    def test_tensor_symmetry(self):
        for cell in all_cells:
            dim = cell.d
            for p in range(1,10):
                for s in (None, True, {(0,1): (1,0)}):
                    # Symmetry dict is invalid for interval cell
                    if isinstance(s,dict) and cell == interval:
                        continue

                    for family in ("Lagrange", "CG", "Discontinuous Lagrange", "DG"):
                        if isinstance(s, dict):
                            element = TensorElement(family, cell, p, shape=(dim,dim), symmetry=s)
                        else:
                            element = TensorElement(family, cell, p, symmetry=s)
                        self.assertEqual(element.value_shape(), (dim,dim))
                        self.assertEqual(element, eval(repr(element)))
                        for i in range(dim):
                            for j in range(dim):
                                c = element.extract_component((i,j))
                                self.assertEqual(c[0], ())

    def test_mixed_tensor_symmetries(self):
        from ufl.algorithms import expand_indices, expand_compounds

        S = FiniteElement('CG', triangle, 1)
        V = VectorElement('CG', triangle, 1)
        T = TensorElement('CG', triangle, 1, symmetry=True)

        # M has dimension 4+1, symmetries are 2->1
        M = T*S
        P = Coefficient(M)
        M = inner(P, P)*dx

        M2 = expand_indices(expand_compounds(M))
        self.assertTrue('[1]' in str(M2))
        self.assertTrue('[2]' not in str(M2))

        # M has dimension 2+(1+4), symmetries are 5->4
        M = V*(S*T)
        P = Coefficient(M)
        M = inner(P, P)*dx

        M2 = expand_indices(expand_compounds(M))
        self.assertTrue('[4]' in str(M2))
        self.assertTrue('[5]' not in str(M2))

    def test_bdm(self):
        for cell in (triangle, tetrahedron):
            dim = cell.d
            element = FiniteElement("BDM", cell, 1)
            self.assertEqual(element.value_shape(), (dim,))
            self.assertEqual(element, eval(repr(element)))

    def test_vector_bdm(self):
        for cell in (triangle, tetrahedron):
            dim = cell.d
            element = VectorElement("BDM", cell, 1)
            self.assertEqual(element.value_shape(), (dim,dim))
            self.assertEqual(element, eval(repr(element)))

    def test_mixed(self):
        for cell in (triangle, tetrahedron):
            dim = cell.d
            velement = VectorElement("CG", cell, 2)
            pelement = FiniteElement("CG", cell, 1)
            TH1 = MixedElement(velement, pelement)
            TH2 = velement * pelement
            self.assertEqual(TH1.value_shape(), (dim+1,))
            self.assertEqual(TH2.value_shape(), (dim+1,))
            self.assertEqual(repr(TH1), repr(TH2))
            self.assertEqual(TH1, eval(repr(TH2)))
            self.assertEqual(TH2, eval(repr(TH1)))

    def test_nested_mixed(self):
        for cell in (triangle, tetrahedron):
            dim = cell.d
            velement = VectorElement("CG", cell, 2)
            pelement = FiniteElement("CG", cell, 1)
            TH1 = MixedElement((velement, pelement), pelement)
            TH2 = velement * pelement * pelement
            self.assertEqual(TH1.value_shape(), (dim+2,))
            self.assertEqual(TH2.value_shape(), (dim+2,))
            self.assertEqual(repr(TH1), repr(TH2))
            self.assertEqual(TH1, eval(repr(TH2)))
            self.assertEqual(TH2, eval(repr(TH1)))

    def test_quadrature_scheme(self):
        for cell in (triangle, tetrahedron):
            for q in (None, 1, 2, 3):
                element = FiniteElement("CG", cell, 1, quad_scheme=q)
                self.assertEqual(element.quadrature_scheme(), q)
                self.assertEqual(element, eval(repr(element)))

    def test_invalid_cell(self):
        from ufl.geometry import as_cell
        for cell in (triangle, as_cell(None)):
            element = FiniteElement("CG", cell, 1)
            self.assertEqual(element, eval(repr(element)))
            #element = VectorElement("CG", cell, 1) # invalid

    def test_invalid_degree(self):
        from ufl.geometry import as_cell
        cell = triangle
        for degree in (1, None):
            element = FiniteElement("CG", cell, degree)
            self.assertEqual(element, eval(repr(element)))
            element = VectorElement("CG", cell, degree)
            self.assertEqual(element, eval(repr(element)))
            print repr(element)

if __name__ == "__main__":
    main()
