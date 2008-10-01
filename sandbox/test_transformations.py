
from ufl import *
from ufl.classes import *
from ufl.algorithms import *


#filename = "../demo/source.ufl"
#forms = load_forms(filename)
#name, form = forms[0]


element = FiniteElement("Lagrange", "triangle", 1)

v = TestFunction(element)
u = TrialFunction(element)
f = Function(element)
g = Function(element)

a = u*(f*(f+g))*(f*(f+g))*v*dx

form = a

formdata = FormData(form)
integral = form.cell_integrals()[0]
integrand = integral.integrand()

assert repr(integrand) == repr(ufl2ufl(integrand))
assert repr(integrand) == repr(ufl2uflcopy(integrand))
assert (integrand is ufl2ufl(integrand))
assert not (integrand is ufl2uflcopy(integrand))

print
print "repr:"
print repr(integrand)
print
print "str:"
print str(integrand)
print
print "ufl2latex:"
print ufl2latex(integrand)
print

integrand = expand_compounds(integrand, 2)
print
print "expand_compounds:"
print
print str(integrand)
print repr(integrand)

original_repr = repr(integrand)

integrand = mark_duplications(integrand)
print
print "mark_duplications:"
print
print str(integrand)
print repr(integrand)

vars = extract_variables(integrand)
print
print "extract_variables:"
print
print "\n".join(str(v) for v in vars)

integrand = strip_variables(integrand)
print
print "strip_variables:"
print
print str(integrand)
print repr(integrand)

assert repr(integrand) == original_repr

rank = 2
num_coefficients = 2
(e, deps, stacks, variable_cache) = split_by_dependencies(integrand, formdata, [(True,True)]*rank, [(True,True)]*num_coefficients)

print
print e
print
print deps
print
print stacks
print
print variable_cache
print

