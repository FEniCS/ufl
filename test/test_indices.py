import pytest

from ufl import (Argument, Coefficient, FunctionSpace, Mesh, TestFunction, TrialFunction, as_matrix, as_tensor,
                 as_vector, cos, dx, exp, i, indices, j, k, l, outer, sin, triangle)
from ufl.classes import IndexSum
from ufl.finiteelement import FiniteElement
from ufl.sobolevspace import H1

# TODO: add more expressions to test as many possible combinations of index notation as feasible...


def test_vector_indices(self):
    element = FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", "triangle", 1, (2, ), (2, ), "identity", H1))
    space = FunctionSpace(domain, element)
    u = Argument(space, 2)
    f = Coefficient(space)
    u[i]*f[i]*dx
    u[j]*f[j]*dx


def test_tensor_indices(self):
    element = FiniteElement("Lagrange", triangle, 1, (2, 2), (2, 2), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", "triangle", 1, (2, ), (2, ), "identity", H1))
    space = FunctionSpace(domain, element)
    u = Argument(space, 2)
    f = Coefficient(space)
    u[i, j]*f[i, j]*dx
    u[j, i]*f[i, j]*dx
    u[j, i]*f[j, i]*dx
    with pytest.raises(BaseException):
        (u[i, i]+f[j, i])*dx


def test_indexed_sum1(self):
    element = FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", "triangle", 1, (2, ), (2, ), "identity", H1))
    space = FunctionSpace(domain, element)
    u = Argument(space, 2)
    f = Coefficient(space)
    a = u[i]+f[i]
    with pytest.raises(BaseException):
        a*dx


def test_indexed_sum2(self):
    element = FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", "triangle", 1, (2, ), (2, ), "identity", H1))
    space = FunctionSpace(domain, element)
    v = Argument(space, 2)
    u = Argument(space, 3)
    f = Coefficient(space)
    a = u[j]+f[j]+v[j]+2*v[j]+exp(u[i]*u[i])/2*f[j]
    with pytest.raises(BaseException):
        a*dx


def test_indexed_sum3(self):
    element = FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", "triangle", 1, (2, ), (2, ), "identity", H1))
    space = FunctionSpace(domain, element)
    u = Argument(space, 2)
    f = Coefficient(space)
    with pytest.raises(BaseException):
        u[i]+f[j]


def test_indexed_function1(self):
    element = FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", "triangle", 1, (2, ), (2, ), "identity", H1))
    space = FunctionSpace(domain, element)
    v = Argument(space, 2)
    u = Argument(space, 3)
    f = Coefficient(space)
    aarg = (u[i]+f[i])*v[i]
    exp(aarg)*dx


def test_indexed_function2(self):
    element = FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", "triangle", 1, (2, ), (2, ), "identity", H1))
    space = FunctionSpace(domain, element)
    v = Argument(space, 2)
    u = Argument(space, 3)
    f = Coefficient(space)
    bfun = cos(f[0])
    left = u[i] + f[i]
    right = v[i] * bfun
    assert len(left.ufl_free_indices) == 1
    assert left.ufl_free_indices[0] == i.count()
    assert len(right.ufl_free_indices) == 1
    assert right.ufl_free_indices[0] == i.count()
    left * right * dx


def test_indexed_function3(self):
    element = FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", "triangle", 1, (2, ), (2, ), "identity", H1))
    space = FunctionSpace(domain, element)
    Argument(space, 2)
    u = Argument(space, 3)
    f = Coefficient(space)
    with pytest.raises(BaseException):
        sin(u[i] + f[i])*dx


def test_vector_from_indices(self):
    element = FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", "triangle", 1, (2, ), (2, ), "identity", H1))
    space = FunctionSpace(domain, element)
    v = TestFunction(space)
    u = TrialFunction(space)

    # legal
    vv = as_vector(u[i], i)
    uu = as_vector(v[j], j)
    w = v + u
    ww = vv + uu
    assert len(vv.ufl_shape) == 1
    assert len(uu.ufl_shape) == 1
    assert len(w.ufl_shape) == 1
    assert len(ww.ufl_shape) == 1


def test_matrix_from_indices(self):
    element = FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", "triangle", 1, (2, ), (2, ), "identity", H1))
    space = FunctionSpace(domain, element)
    v = TestFunction(space)
    u = TrialFunction(space)

    A = as_matrix(u[i]*v[j], (i, j))
    B = as_matrix(v[k]*v[k]*u[i]*v[j], (j, i))
    C = A + A
    C = B + B
    D = A + B
    assert len(A.ufl_shape) == 2
    assert len(B.ufl_shape) == 2
    assert len(C.ufl_shape) == 2
    assert len(D.ufl_shape) == 2


def test_vector_from_list(self):
    element = FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", "triangle", 1, (2, ), (2, ), "identity", H1))
    space = FunctionSpace(domain, element)
    v = TestFunction(space)
    u = TrialFunction(space)

    # create vector from list
    vv = as_vector([u[0], v[0]])
    ww = vv + vv
    assert len(vv.ufl_shape) == 1
    assert len(ww.ufl_shape) == 1


def test_matrix_from_list(self):
    element = FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", "triangle", 1, (2, ), (2, ), "identity", H1))
    space = FunctionSpace(domain, element)
    v = TestFunction(space)
    u = TrialFunction(space)

    # create matrix from list
    A = as_matrix([[u[0], u[1]], [v[0], v[1]]])
    # create matrix from indices
    B = as_matrix((v[k]*v[k]) * u[i]*v[j], (j, i))
    # Test addition
    C = A + A
    C = B + B
    D = A + B
    assert len(A.ufl_shape) == 2
    assert len(B.ufl_shape) == 2
    assert len(C.ufl_shape) == 2
    assert len(D.ufl_shape) == 2


def test_tensor(self):
    element = FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", "triangle", 1, (2, ), (2, ), "identity", H1))
    space = FunctionSpace(domain, element)
    v = TestFunction(space)
    u = TrialFunction(space)
    f = Coefficient(space)
    g = Coefficient(space)

    # define the components of a fourth order tensor
    Cijkl = u[i]*v[j]*f[k]*g[l]
    assert len(Cijkl.ufl_shape) == 0
    assert set(Cijkl.ufl_free_indices) == {i.count(), j.count(), k.count(), l.count()}

    # make it a tensor
    C = as_tensor(Cijkl, (i, j, k, l))
    assert len(C.ufl_shape) == 4
    self.assertSameIndices(C, ())

    # get sub-matrix
    A = C[:, :, 0, 0]
    assert len(A.ufl_shape) == 2
    self.assertSameIndices(A, ())
    A = C[:, :, i, j]
    assert len(A.ufl_shape) == 2
    assert set(A.ufl_free_indices) == {i.count(), j.count()}

    # legal?
    vv = as_vector([u[i], v[i]])
    f[i]*vv  # this is well defined: ww = sum_i <f_i*u_i, f_i*v_i>

    # illegal
    with pytest.raises(BaseException):
        as_vector([u[i], v[j]])

    # illegal
    with pytest.raises(BaseException):
        as_matrix([[u[0], u[1]], [v[0]]])


def test_indexed(self):
    element = FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", "triangle", 1, (2, ), (2, ), "identity", H1))
    space = FunctionSpace(domain, element)
    v = TestFunction(space)
    u = TrialFunction(space)
    Coefficient(space)
    i, j, k, l = indices(4)  # noqa: E741

    a = v[i]
    self.assertSameIndices(a, (i,))

    a = outer(v, u)[i, j]
    self.assertSameIndices(a, (i, j))

    a = outer(v, u)[i, i]
    self.assertSameIndices(a, ())
    self.assertIsInstance(a, IndexSum)


def test_spatial_derivative(self):
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1, (2, ), (2, ), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", cell, 1, (2, ), (2, ), "identity", H1))
    space = FunctionSpace(domain, element)
    v = TestFunction(space)
    u = TrialFunction(space)
    i, j, k, l = indices(4)  # noqa: E741
    d = cell.geometric_dimension()

    a = v[i].dx(i)
    self.assertSameIndices(a, ())
    self.assertIsInstance(a, IndexSum)
    assert a.ufl_shape == ()

    a = v[i].dx(j)
    self.assertSameIndices(a, (i, j))
    self.assertNotIsInstance(a, IndexSum)
    assert a.ufl_shape == ()

    a = (v[i]*u[j]).dx(i, j)
    self.assertSameIndices(a, ())
    self.assertIsInstance(a, IndexSum)
    assert a.ufl_shape == ()

    a = v.dx(i, j)
    # self.assertSameIndices(a, (i,j))
    assert set(a.ufl_free_indices) == {j.count(), i.count()}
    self.assertNotIsInstance(a, IndexSum)
    assert a.ufl_shape == (d,)

    a = v[i].dx(0)
    self.assertSameIndices(a, (i,))
    self.assertNotIsInstance(a, IndexSum)
    assert a.ufl_shape == ()

    a = (v[i]*u[j]).dx(0, 1)
    assert set(a.ufl_free_indices) == {i.count(), j.count()}
    self.assertNotIsInstance(a, IndexSum)
    assert a.ufl_shape == ()

    a = v.dx(i)[i]
    self.assertSameIndices(a, ())
    self.assertIsInstance(a, IndexSum)
    assert a.ufl_shape == ()


def test_renumbering(self):
    pass
