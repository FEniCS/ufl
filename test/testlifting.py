
if __name__ == "__main__":
    from ufl import triangle, FiniteElement, VectorElement, dE, Coefficient, LiftingCoefficient, LiftingOperator, TestFunction, dot, dx
    from ufl.algorithms import tree_format
    cell = triangle
    u_space = FiniteElement("DG", cell, 1)
    l_space = VectorElement("DG", cell, 0)
    R = LiftingCoefficient(l_space)
    r = LiftingOperator(l_space)

    u = Coefficient(u_space)
    v = TestFunction(u_space)

    a = dot(r(u), r(v))*dE
    print 
    print str(a)
    print repr(a)
    print tree_format(a)

    a = dot(R(u), R(v))*dE
    print 
    print str(a)
    print repr(a)
    print tree_format(a)

