
from ufl import *

cell = triangle

element = FiniteElement("CG", cell, 1)
v = TestFunction(element)
u = TrialFunction(element)
f = Function(element)

velement = VectorElement("CG", cell, 1)
vv = TestFunction(velement)
vu = TrialFunction(velement)
vf = Function(velement)

telement = TensorElement("CG", cell, 1)
tv = TestFunction(telement)
tu = TrialFunction(telement)
tf = Function(telement)

