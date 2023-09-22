from ufl import (Identity, Mesh, SpatialCoordinate, as_matrix, as_ufl, as_vector, elem_div, elem_mult, elem_op, sin,
                 tetrahedron, triangle)
from ufl.classes import ComplexValue, Division, FloatValue, IntValue
from ufl.finiteelement import FiniteElement
from ufl.pull_back import identity_pull_back
from ufl.sobolevspace import H1


def test_scalar_casting(self):
    f = as_ufl(2.0)
    r = as_ufl(4)
    c = as_ufl(1 + 2j)
    self.assertIsInstance(f, FloatValue)
    self.assertIsInstance(r, IntValue)
    self.assertIsInstance(c, ComplexValue)
    assert float(f) == 2.0
    assert int(r) == 4
    assert complex(c) == 1.0 + 2.0j


def test_ufl_float_division(self):
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), identity_pull_back, H1))
    d = SpatialCoordinate(domain)[0] / 10.0  # TODO: Use mock instead of x
    self.assertIsInstance(d, Division)


def test_float_ufl_division(self):
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), identity_pull_back, H1))
    d = 3.14 / SpatialCoordinate(domain)[0]  # TODO: Use mock instead of x
    self.assertIsInstance(d, Division)


def test_float_division(self):
    d = as_ufl(20.0) / 10.0
    self.assertIsInstance(d, FloatValue)
    assert float(d) == 2.0


def test_int_division(self):
    # UFL treats all divisions as true division
    d = as_ufl(40) / 7
    self.assertIsInstance(d, FloatValue)
    assert float(d) == 40.0 / 7.0
    # self.assertAlmostEqual(float(d), 40 / 7.0, 15)


def test_float_int_division(self):
    d = as_ufl(20.0) / 5
    self.assertIsInstance(d, FloatValue)
    assert float(d) == 4.0


def test_floor_division_fails(self):
    f = as_ufl(2.0)
    r = as_ufl(4)
    s = as_ufl(5)
    self.assertRaises(NotImplementedError, lambda: r // 4)
    self.assertRaises(NotImplementedError, lambda: r // s)
    self.assertRaises(NotImplementedError, lambda: f // s)


def test_elem_mult(self):
    self.assertEqual(int(elem_mult(2, 3)), 6)

    v = as_vector((1, 2, 3))
    u = as_vector((4, 5, 6))
    self.assertEqual(elem_mult(v, u), as_vector((4, 10, 18)))


def test_elem_mult_on_matrices(self):
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), identity_pull_back, H1))

    A = as_matrix(((1, 2), (3, 4)))
    B = as_matrix(((4, 5), (6, 7)))
    self.assertEqual(elem_mult(A, B), as_matrix(((4, 10), (18, 28))))

    x, y = SpatialCoordinate(domain)
    A = as_matrix(((x, y), (3, 4)))
    B = as_matrix(((4, 5), (y, x)))
    self.assertEqual(elem_mult(A, B), as_matrix(((4*x, 5*y), (3*y, 4*x))))

    x, y = SpatialCoordinate(domain)
    A = as_matrix(((x, y), (3, 4)))
    B = Identity(2)
    self.assertEqual(elem_mult(A, B), as_matrix(((x, 0), (0, 4))))


def test_elem_div(self):
    domain = Mesh(FiniteElement("Lagrange", tetrahedron, 1, (3, ), (3, ), identity_pull_back, H1))
    x, y, z = SpatialCoordinate(domain)
    A = as_matrix(((x, y, z), (3, 4, 5)))
    B = as_matrix(((7, 8, 9), (z, x, y)))
    self.assertEqual(elem_div(A, B), as_matrix(((x/7, y/8, z/9), (3/z, 4/x, 5/y))))


def test_elem_op(self):
    domain = Mesh(FiniteElement("Lagrange", tetrahedron, 1, (3, ), (3, ), identity_pull_back, H1))
    x, y, z = SpatialCoordinate(domain)
    A = as_matrix(((x, y, z), (3, 4, 5)))
    self.assertEqual(elem_op(sin, A), as_matrix(((sin(x), sin(y), sin(z)),
                                                 (sin(3), sin(4), sin(5)))))
    self.assertEqual(elem_op(sin, A).dx(0).ufl_shape, (2, 3))
