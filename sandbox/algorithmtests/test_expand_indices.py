
from ufl import *
from ufl.classes import *
from ufl.algorithms import *

def test1():
    #name = "../../demo/StiffnessAD.ufl"
    name = "../../demo/Constant.ufl"
    forms = load_forms(name)

    for f in forms:
        fd = f.form_data()
        g = fd.form
        print 
        print fd.name
        print 
        print str(f)
        print 
        print str(g)
        print 
        print str(expand_indices(g))
        print 

def test2():
    element = FiniteElement("Lagrange", triangle, 2)
    v = TestFunction(element)
    u = TrialFunction(element)
    
    a = div(grad(v))*u*dx
    print tree_format(a)
    a = expand_derivatives(a)
    print tree_format(a)
    a = expand_indices(a)
    print tree_format(a)

if __name__ == "__main__":
    test2()

