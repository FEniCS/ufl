
from ufl import *
from ufl.algorithms import *

e = FiniteElement("DG", triangle, 1)
f = Function(e)
v = TestFunction(e)
L = f*v*dS

print [propagate_restrictions(itg.integrand()) for itg in L.form_data().form.integrals()]

