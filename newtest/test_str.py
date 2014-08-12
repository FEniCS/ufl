#!/usr/bin/env py.test

import pytest

from ufl import *
from ufl.classes import *


def test_str_int_value(self):
    self.assertEqual(str(as_ufl(3)), "3")

def test_str_float_value(self):
    self.assertEqual(str(as_ufl(3.14)), "3.14")

def test_str_zero(self):
    x = SpatialCoordinate(triangle)
    self.assertEqual(str(as_ufl(0)), "0")
    self.assertEqual(str(0*x), "(0<(2,), ()>)") # TODO: Not very nice...
    self.assertEqual(str(0*x*x[Index(42)]), "(0<(2,), (42,)>)") # TODO: Not very nice...

def test_str_index(self):
    self.assertEqual(str(Index(3)), "i_3")
    self.assertEqual(str(Index(42)), "i_{42}")


def test_str_coordinate(self):
    self.assertEqual(str(SpatialCoordinate(triangle)), "x")
    self.assertEqual(str(SpatialCoordinate(triangle)[0]), "(x)[0]") # FIXME: Get rid of extra ()

def test_str_normal(self):
    self.assertEqual(str(FacetNormal(triangle)), "n")
    self.assertEqual(str(FacetNormal(triangle)[0]), "(n)[0]") # FIXME: Get rid of extra ()

def test_str_circumradius(self):
    self.assertEqual(str(Circumradius(triangle)), "circumradius")

#def test_str_cellsurfacearea(self):
#    self.assertEqual(str(CellSurfaceArea(triangle)), "surfacearea")

def test_str_facetarea(self):
    self.assertEqual(str(FacetArea(triangle)), "facetarea")

def test_str_volume(self):
    self.assertEqual(str(CellVolume(triangle)), "volume")


def test_str_scalar_argument(self):
    v = TestFunction(FiniteElement("CG", triangle, 1))
    u = TrialFunction(FiniteElement("CG", triangle, 1))
    self.assertEqual(str(v), "v_0")
    self.assertEqual(str(u), "v_1")

#def test_str_vector_argument(self): # FIXME

#def test_str_scalar_coefficient(self): # FIXME

#def test_str_vector_coefficient(self): # FIXME


def test_str_list_vector(self):
    x, y, z = SpatialCoordinate(tetrahedron)
    v = as_vector((x, y, z))
    self.assertEqual(str(v), "[%s, %s, %s]" % (x, y, z))

def test_str_list_vector_with_zero(self):
    x, y, z = SpatialCoordinate(tetrahedron)
    v = as_vector((x, 0, 0))
    self.assertEqual(str(v), "[%s, 0, 0]" % (x,))

def test_str_list_matrix(self):
    x, y = SpatialCoordinate(triangle)
    v = as_matrix(((2*x, 3*y),
                   (4*x, 5*y)))
    a = str(2*x)
    b = str(3*y)
    c = str(4*x)
    d = str(5*y)
    self.assertEqual(str(v), "[\n  [%s, %s],\n  [%s, %s]\n]" % (a, b, c, d))

def test_str_list_matrix_with_zero(self):
    x, y = SpatialCoordinate(triangle)
    v = as_matrix(((2*x, 3*y),
                   (0, 0)))
    a = str(2*x)
    b = str(3*y)
    c = str(as_vector((0, 0)))
    self.assertEqual(str(v), "[\n  [%s, %s],\n%s\n]" % (a, b, c))

# FIXME: Add more tests for tensors collapsing
#        partly or completely into Zero!
