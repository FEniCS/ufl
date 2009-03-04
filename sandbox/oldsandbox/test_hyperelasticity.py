
from ufl import *
from ufl.algorithms import *

forms = load_forms("../demo/HyperElasticity.ufl")
for orig_form in forms:
    fd = orig_form.form_data()
    form = fd.form
    name = fd.name
    itg = form.cell_integrals()[0]
    x = itg._integrand
    #print repr(x)
    #print str(x)
    y = ufl2ufl(x)
    #print ufl2latex(x)
    #print expand_compounds(x)

