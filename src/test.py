from ufl import *
from ufl.utilities import * 

def test_iterators():
    polygon = "triangle"
    fe1 = FiniteElement("Lagrange", polygon, 1)
    fe2 = FiniteElement("Lagrange", polygon, 2)
    a = BasisFunction(fe1)
    b = Constant(polygon, "b")
    c = BasisFunction(fe2)
    d = Constant(fe1, "d")
    e = BasisFunction(fe2)
    g = Function(fe2, "b")
    f = exp(a+b*c/d)**g-e
    print f
    print ""
    print list(ufl_objs(f))
    print ""
    print "iter_depth_first(f):"
    print list(iter_depth_first(f))
    print ""
    print "iter_width_first(f):"
    print list(iter_width_first(f))
    print ""
    print "iter_classes(f):"
    print list(iter_classes(f))
    print ""
    print "iter_classes(f*dx):"
    print list(iter_classes(f*dx))
    print ""
    print "f has a UFLFunction: ", any(c is UFLFunction for c in iter_classes(f))
    print ""
    print "iter_basisfunctions(f):"
    print list(iter_basisfunctions(f))
    print ""
    print "iter_coefficients(f):"
    print list(iter_coefficients(f))
    print ""
    print "iter_elements(f):"
    print list(iter_elements(f))
    print ""
test_iterators()

def test_integrals():
    a = 1*dx
    b = 2*ds
    c = 1*dx0 + 2*dx1 + 3*ds0 + 4*ds1

def test_elements():
    selement = FiniteElement("Lagrange", "triangle", 1)
    velement = VectorElement("Lagrange", "triangle", 1)
    telement = TensorElement("Lagrange", "triangle", 1)

def test_functions():
    selement = FiniteElement("Lagrange", "triangle", 1)
    velement = VectorElement("Lagrange", "triangle", 1)
    telement = TensorElement("Lagrange", "triangle", 1)
    qelement = QuadratureElement("cell", "triangle")

    for element in (selement, velement, telement, qelement):
        v = TestFunction(element)
        u = TrialFunction(element)
        f = Function(element)
        c = Constant(element.polygon)

def test_contains():
    element = FiniteElement("Lagrange", "triangle", 1)
    v = TestFunction(element)
    u = TrialFunction(element)
    f = Function(element)

    a = f*(u*v)*dx
    print "a =", a
    print "f in a", f in a
    print "u in a", u in a
    print "v in a", v in a
    print "f*u in a", f*u in a
    print "u*v in a", u*v in a
    print "f*(u*v) in a", f*(u*v) in a
    b = 2*ds
    c = 1*dx0 + 2*dx1 + 3*ds0 + 4*ds1

    
def test_basic_algebra():
    element = FiniteElement("Lagrange", "triangle", 1)
    v = TestFunction(element)
    u = TrialFunction(element)
    f = Function(element)
    c = Constant(element.polygon)
    
    a = v + u
    print a
    
    s = v - u
    print s
    
    m = v * u
    print m
    
    d = v / u
    print d

    p = u**3
    print p


if __name__ == "__main__":

    test_elements()
    test_functions()
    test_integrals()
    test_basic_algebra()
    test_contains()

    print "TODO: clean up tests"

    I = Id()
    
    Dv = grad(v)
    Du = grad(u)
    F  = I + Du.T()
    
    J = det(F)
    
    C = F.T() * F
    E = (C - I) / 2
    
    Ef = A.T() * E * A
    
    # TODO: I want to do this:
    #Q   = Ef[0,0]**2
    Q = Ef**2
    psi = exp(Q) - (1-J)*ln(J)
    
    # TODO: I want to do this:
    #S = diff(psi, E)
    S = psi*E
    
    a = inner(F*S, grad(v)) * dx
    print a
    
    b = inner(grad(u), grad(v)) * dx1
    print b
    
    c = inner(u, v) * dx2
    print c
    
    print (b+c)
    
    x = Symbol("x")
    y = Symbol("y")


    def test_traverse():
        ufl = grad(a) - div(m) * (wedge(m) / a*s*m)
        
        def func(o):
            print o
        
        print ""
        print "ufl:"
        print ufl
        print ""
        print "traverse_width_first:"
        traverse_width_first(func, ufl)
        print ""
        print "traverse_depth_first:"
        traverse_depth_first(func, ufl)
        print ""
        
