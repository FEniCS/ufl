
from ufl import *
from ufl.algorithms import *

formdatas = load_forms("../demo/HyperElasticity.ufl")
for fd in formdatas:
    form = fd.form
    name = fd.name
    itg = form.cell_integrals()[0]
    x = itg._integrand
    #print repr(x)
    #print str(x)
    #print ufl2ufl(x)
    #print ufl2latex(x)
    #print expand_compounds(x)

