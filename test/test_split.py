__authors__ = "Martin Sandve Alnæs"
__date__ = "2009-03-14 -- 2009-03-14"

from ufl import (Coefficient, FiniteElement, FunctionSpace, Mesh, MixedElement, TensorElement, TestFunction,
                 VectorElement, as_vector, product, split, triangle)


def test_split(self):
    cell = triangle
    domain = Mesh(VectorElement("Lagrange", cell, 1))
    d = cell.geometric_dimension()
    f = FiniteElement("CG", cell, 1)
    v = VectorElement("CG", cell, 1)
    w = VectorElement("CG", cell, 1, dim=d+1)
    t = TensorElement("CG", cell, 1)
    s = TensorElement("CG", cell, 1, symmetry=True)
    r = TensorElement("CG", cell, 1, symmetry={(1, 0): (0, 1)}, shape=(d, d))
    m = MixedElement(f, v, w, t, s, r)

    f_space = FunctionSpace(domain, f)
    v_space = FunctionSpace(domain, v)
    w_space = FunctionSpace(domain, w)
    t_space = FunctionSpace(domain, t)
    s_space = FunctionSpace(domain, s)
    r_space = FunctionSpace(domain, r)
    m_space = FunctionSpace(domain, m)

    # Check that shapes of all these functions are correct:
    assert () == Coefficient(f_space).ufl_shape
    assert (d,) == Coefficient(v_space).ufl_shape
    assert (d+1,) == Coefficient(w_space).ufl_shape
    assert (d, d) == Coefficient(t_space).ufl_shape
    assert (d, d) == Coefficient(s_space).ufl_shape
    assert (d, d) == Coefficient(r_space).ufl_shape
    # sum of value sizes, not accounting for symmetries:
    assert (3*d*d + 2*d + 2,) == Coefficient(m_space).ufl_shape

    # Shapes of subelements are reproduced:
    g = Coefficient(m_space)
    s, = g.ufl_shape
    for g2 in split(g):
        s -= product(g2.ufl_shape)
    assert s == 0

    # Mixed elements of non-scalar subelements are flattened
    v2 = MixedElement(v, v)
    m2 = MixedElement(t, t)
    v2_space = FunctionSpace(domain, v2)
    m2_space = FunctionSpace(domain, m2)
    # assert d == 2
    # assert (2,2) == Coefficient(v2_space).ufl_shape
    assert (d+d,) == Coefficient(v2_space).ufl_shape
    assert (2*d*d,) == Coefficient(m2_space).ufl_shape

    # Split simple element
    t = TestFunction(f_space)
    assert split(t) == (t,)

    # Split twice on nested mixed elements gets
    # the innermost scalar subcomponents
    t = TestFunction(FunctionSpace(domain, f*v))
    assert split(t) == (t[0], as_vector((t[1], t[2])))
    assert split(split(t)[1]) == (t[1], t[2])
    t = TestFunction(FunctionSpace(domain, f*(f*v)))
    assert split(t) == (t[0], as_vector((t[1], t[2], t[3])))
    assert split(split(t)[1]) == (t[1], as_vector((t[2], t[3])))
    t = TestFunction(FunctionSpace(domain, (v*f)*(f*v)))
    assert split(t) == (as_vector((t[0], t[1], t[2])),
                        as_vector((t[3], t[4], t[5])))
    assert split(split(t)[0]) == (as_vector((t[0], t[1])), t[2])
    assert split(split(t)[1]) == (t[3], as_vector((t[4], t[5])))
    assert split(split(split(t)[0])[0]) == (t[0], t[1])
    assert split(split(split(t)[0])[1]) == (t[2],)
    assert split(split(split(t)[1])[0]) == (t[3],)
    assert split(split(split(t)[1])[1]) == (t[4], t[5])
