a = [('Argument', 160), ('Label', 400), ('Variable', 440), ('Coefficient', 832), ('Division', 62128), ('Exp', 99264), ('FloatValue', 542592), ('Power', 1435984), ('IntValue', 5676696), ('Sum', 10477520), ('Product', 14031360), ('SpatialDerivative', 16739448), ('Indexed', 21631344), ('MultiIndex', 27601776)]

print((sum(x[1] for x in a)))

import ufl

n = ufl.triangle.n
v = n[ufl.i]
ii = v.operands()[1]
print((type(ii)))
import sys
print((sys.getsizeof(ii)))


from ufl import *
V = VectorElement("CG", tetrahedron, 1)
u = Coefficient(V)
W = FiniteElement("CG", triangle, 1)
w = Coefficient(W)
print((sys.getsizeof(V)))
print((sys.getsizeof(u)))
print((sys.getsizeof(W)))
print((sys.getsizeof(w)))

print()
for cl in ufl.classes.terminal_classes:
    print((cl.__name__, sys.getsizeof(cl)))


