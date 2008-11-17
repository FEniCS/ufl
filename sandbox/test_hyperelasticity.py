
from ufl import *
from ufl.algorithms import *

forms = load_forms("../demo/hyperelasticity.ufl_")
for (name, form) in forms:
    itg = form.cell_integrals()[0]
    x = itg._integrand
    print repr(x)
    print str(x)
    print ufl2ufl(x)
    print ufl2latex(x)
    print expand_compounds(x)

