
from dolfin import cpp

import ufl

class SomeFunction(object):
    def __init__(self, value):
        self._a = value
        self._b = 2*value
    
    def __getattr__(self,name):
        return self.__dict__[name]

def init(self, V, value):
    ufl.Function.__init__(self, V)
    cpp.Function.__init__(self)
    self._V = V

def Function_factory():
    return type("MyFunction", (ufl.Function, cpp.Function, Function), { "__init__": init })

class Function(object):
    def __new__(cls, V, value):
        # If the __new__ function is called because we are instantiating a sub class
        # of Function. Instantiate the class directly using objects __new__
        if cls.__name__ != "Function":
            print("It's alive")
            return object.__new__(cls)
        return object.__new__(Function_factory())

V = ufl.FiniteElement("CG", "triangle", 1)
v = ufl.TestFunction(V)
f = Function(V, 3)
print((f._element))
print((f._a))

L = f*v*ufl.dx
print(L)

