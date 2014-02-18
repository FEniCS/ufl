#!/usr/bin/env python

"""
Test the is_cellwise_constant function on all relevant terminal types.
"""

from ufltestcase import UflTestCase, main
from ufl import *


class TestCellwiseConstantTerminals(UflTestCase):

    def setUp(self):
        super(TestCellwiseConstantTerminals, self).setUp()

        self.all_cells = [
            cell1D,
            interval,
            cell2D,
            triangle,
            quadrilateral,
            cell3D,
            tetrahedron,
            hexahedron,
            ]
        self.domains = [Domain(cell) for cell in self.all_cells]
        self.domains_with_linear_coordinates = []
        for D in self.domains:
            V = VectorElement("CG", D, 1)
            x = Coefficient(V)
            E = Domain(x)
            self.domains_with_linear_coordinates.append(E)
        self.domains_with_quadratic_coordinates = []
        for D in self.domains:
            V = VectorElement("CG", D, 2)
            x = Coefficient(V)
            E = Domain(x)
            self.domains_with_quadratic_coordinates.append(E)

        self.affine_cells = [
            interval,
            triangle,
            tetrahedron,
            ]
        self.affine_domains = [Domain(cell) for cell in self.affine_cells]
        self.affine_domains_with_linear_coordinates = []
        for D in self.affine_domains:
            V = VectorElement("CG", D, 1)
            x = Coefficient(V)
            E = Domain(x)
            self.affine_domains_with_linear_coordinates.append(E)

        self.affine_facet_cells = [
            interval,
            cell1D,
            triangle,
            quadrilateral,
            tetrahedron,
            ]
        self.affine_facet_domains = [Domain(cell) for cell in self.affine_facet_cells]
        self.affine_facet_domains_with_linear_coordinates = []
        for D in self.affine_facet_domains:
            V = VectorElement("CG", D, 1)
            x = Coefficient(V)
            E = Domain(x)
            self.affine_facet_domains_with_linear_coordinates.append(E)

        self.nonaffine_cells = [
            cell2D,
            quadrilateral,
            cell3D,
            hexahedron,
            ]
        self.nonaffine_domains = [Domain(cell) for cell in self.nonaffine_cells]
        self.nonaffine_domains_with_linear_coordinates = []
        for D in self.nonaffine_domains:
            V = VectorElement("CG", D, 1)
            x = Coefficient(V)
            E = Domain(x)
            self.nonaffine_domains_with_linear_coordinates.append(E)

        self.nonaffine_facet_cells = [
            cell2D,
            cell3D,
            hexahedron,
            ]
        self.nonaffine_facet_domains = [Domain(cell) for cell in self.nonaffine_facet_cells]
        self.nonaffine_facet_domains_with_linear_coordinates = []
        for D in self.nonaffine_facet_domains:
            V = VectorElement("CG", D, 1)
            x = Coefficient(V)
            E = Domain(x)
            self.nonaffine_facet_domains_with_linear_coordinates.append(E)

    def test_always_cellwise_constant_geometric_quantities(self):
        "Test geometric quantities that are always constant over a cell."
        domains = []
        domains += self.domains
        domains += self.domains_with_linear_coordinates
        domains += self.domains_with_quadratic_coordinates
        for D in domains:
            e = CellVolume(D)
            self.assertTrue(e.is_cellwise_constant())
            e = Circumradius(D)
            self.assertTrue(e.is_cellwise_constant())
            e = FacetArea(D)
            self.assertTrue(e.is_cellwise_constant())
            e = MinFacetEdgeLength(D)
            self.assertTrue(e.is_cellwise_constant())
            e = MaxFacetEdgeLength(D)
            self.assertTrue(e.is_cellwise_constant())

    def test_coordinates_never_cellwise_constant(self):
        domains = []
        domains += self.domains
        domains += self.domains_with_linear_coordinates
        domains += self.domains_with_quadratic_coordinates
        for D in domains:
            e = SpatialCoordinate(D)
            self.assertFalse(e.is_cellwise_constant())
            e = LocalCoordinate(D)
            self.assertFalse(e.is_cellwise_constant())

        # The only exception here:
        D = Domain(Cell("vertex", 3))
        self.assertEqual(D.cell().cellname(), "vertex")
        e = SpatialCoordinate(D)
        self.assertTrue(e.is_cellwise_constant())
        e = LocalCoordinate(D)
        self.assertTrue(e.is_cellwise_constant())

    def test_mappings_are_cellwise_constant_only_on_linear_affine_cells(self):
        domains = []
        domains += self.affine_domains
        domains += self.affine_domains_with_linear_coordinates
        for D in domains:
            e = Jacobian(D)
            self.assertTrue(e.is_cellwise_constant())
            e = JacobianDeterminant(D)
            self.assertTrue(e.is_cellwise_constant())
            e = JacobianInverse(D)
            self.assertTrue(e.is_cellwise_constant())
            e = FacetJacobian(D)
            self.assertTrue(e.is_cellwise_constant())
            e = FacetJacobianDeterminant(D)
            self.assertTrue(e.is_cellwise_constant())
            e = FacetJacobianInverse(D)
            self.assertTrue(e.is_cellwise_constant())

        domains = []
        domains += self.nonaffine_domains
        domains += self.nonaffine_domains_with_linear_coordinates
        domains += self.domains_with_quadratic_coordinates
        for D in domains:
            e = Jacobian(D)
            self.assertFalse(e.is_cellwise_constant())
            e = JacobianDeterminant(D)
            self.assertFalse(e.is_cellwise_constant())
            e = JacobianInverse(D)
            self.assertFalse(e.is_cellwise_constant())
            e = FacetJacobian(D)
            self.assertFalse(e.is_cellwise_constant())
            e = FacetJacobianDeterminant(D)
            self.assertFalse(e.is_cellwise_constant())
            e = FacetJacobianInverse(D)
            self.assertFalse(e.is_cellwise_constant())

    def test_facetnormal_sometimes_cellwise_constant(self):
        domains = []
        domains += self.affine_facet_domains
        domains += self.affine_facet_domains_with_linear_coordinates
        for D in domains:
            e = FacetNormal(D)
            self.assertTrue(e.is_cellwise_constant())

        domains = []
        domains += self.nonaffine_facet_domains
        domains += self.nonaffine_facet_domains_with_linear_coordinates
        domains += self.domains_with_quadratic_coordinates
        for D in domains:
            e = FacetNormal(D)
            self.assertFalse(e.is_cellwise_constant())

# Don't touch these lines, they allow you to run this file directly
if __name__ == "__main__":
    main()

