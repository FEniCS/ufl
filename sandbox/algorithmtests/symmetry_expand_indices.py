
from ufl import *
from ufl.algorithms import *

e = TensorElement("CG", triangle, 1, symmetry=True)
f = Function(e)
v = TestFunction(e)
a = inner(f, v)*dx

fd = a.form_data()
b = fd.form
c = expand_indices(b)
print((tree_format(c)))
print((str(c)))

#   (v_0)[0, 0] * (w_0)[0, 0] 
# + (v_0)[0, 1] * (w_0)[0, 1] 
# + (v_0)[0, 1] * (w_0)[0, 1] 
# + (v_0)[1, 1] * (w_0)[1, 1]

