__authors__ = "Martin Sandve Alnæs"
__date__ = "2009-03-14 -- 2009-03-14"

from ufl import Coefficient, FunctionSpace, Mesh, TestFunction, as_vector, product, split, triangle
from ufl.finiteelement import FiniteElement, MixedElement
from ufl.sobolevspace import H1


def test_split(self):
    cell = triangle
    d = cell.geometric_dimension()
    domain = Mesh(FiniteElement("Lagrange", cell, 1, (d, ), (d, ), "identity", H1))
    f = FiniteElement("Lagrange", cell, 1, (), (), "identity", H1)
    v = FiniteElement("Lagrange", cell, 1, (d, ), (d, ), "identity", H1,
                      sub_elements=[f for _ in range(d)])
    w = FiniteElement("Lagrange", cell, 1, (d+1, ), (d+1, ), "identity", H1,
                      sub_elements=[f for _ in range(d + 1)])
    t = FiniteElement("Lagrange", cell, 1, (d, d), (d, d), "identity", H1,
                      sub_elements=[f for _ in range(d ** 2)])
    s = FiniteElement("Lagrange", cell, 1, (2, 2), (3, ), "identity", H1,
                      component_map={(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 2},
                      sub_elements=[f for _ in range(3)])
    m = MixedElement([f, v, w, t, s, s])

    f_space = FunctionSpace(domain, f)
    v_space = FunctionSpace(domain, v)
    w_space = FunctionSpace(domain, w)
    t_space = FunctionSpace(domain, t)
    s_space = FunctionSpace(domain, s)
    m_space = FunctionSpace(domain, m)

    # Check that shapes of all these functions are correct:
    assert () == Coefficient(f_space).ufl_shape
    assert (d,) == Coefficient(v_space).ufl_shape
    assert (d+1,) == Coefficient(w_space).ufl_shape
    assert (d, d) == Coefficient(t_space).ufl_shape
    assert (d, d) == Coefficient(s_space).ufl_shape
    # sum of value sizes, not accounting for symmetries:
    assert (3*d*d + 2*d + 2,) == Coefficient(m_space).ufl_shape

    # Shapes of subelements are reproduced:
    g = Coefficient(m_space)
    s, = g.ufl_shape
    for g2 in split(g):
        s -= product(g2.ufl_shape)
    assert s == 0

    # Mixed elements of non-scalar subelements are flattened
    v2 = MixedElement([v, v])
    m2 = MixedElement([t, t])
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
    t = TestFunction(FunctionSpace(domain, MixedElement([f, v])))
    assert split(t) == (t[0], as_vector((t[1], t[2])))
    assert split(split(t)[1]) == (t[1], t[2])
    t = TestFunction(FunctionSpace(domain, MixedElement([f, [f, v]])))
    assert split(t) == (t[0], as_vector((t[1], t[2], t[3])))
    assert split(split(t)[1]) == (t[1], as_vector((t[2], t[3])))
    t = TestFunction(FunctionSpace(domain, MixedElement([[v, f], [f, v]])))
    assert split(t) == (as_vector((t[0], t[1], t[2])),
                        as_vector((t[3], t[4], t[5])))
    assert split(split(t)[0]) == (as_vector((t[0], t[1])), t[2])
    assert split(split(t)[1]) == (t[3], as_vector((t[4], t[5])))
    assert split(split(split(t)[0])[0]) == (t[0], t[1])
    assert split(split(split(t)[0])[1]) == (t[2],)
    assert split(split(split(t)[1])[0]) == (t[3],)
    assert split(split(split(t)[1])[1]) == (t[4], t[5])
