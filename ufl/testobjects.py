"Some premade objects useful for quick testing."
from ufl import *

cell = triangle

element = FiniteElement("CG", cell, 1)
v = TestFunction(element)
u = TrialFunction(element)
f = Coefficient(element)

velement = VectorElement("CG", cell, 1)
vv = TestFunction(velement)
vu = TrialFunction(velement)
vf = Coefficient(velement)

telement = TensorElement("CG", cell, 1)
tv = TestFunction(telement)
tu = TrialFunction(telement)
tf = Coefficient(telement)

