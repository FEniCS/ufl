
import ufl

class SomeFunction(object):
    def __init__(self):
        pass

    def __getattr__(self, name):
        print "SomeFunction.getattr:", name
        return object.__dict__[name]

class MyFunction(ufl.Function, SomeFunction):
    def __init__(self, V):
        ufl.Function.__init__(self, V)

    def __getattr__(self, name):
        print "MyFunction.getattr:", name
        return object.__dict__[name]

V = ufl.FiniteElement("CG", "triangle", 1)
v = ufl.TestFunction(V)
f = MyFunction(V)
print f._element

a = f*v*ufl.dx
print a

print f._element
print f.foo
