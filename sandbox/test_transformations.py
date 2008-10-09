
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

rank = len(formdata.basisfunctions)
num_coefficients = len(formdata.coefficients)

from ufl.algorithms.dependencies import DependencySet

basisfunction_deps = []
for i in range(rank):
    # TODO: More dependencies depending on element
    bfs = tuple(i == j for j in range(rank))
    fs = (False,)*num_coefficients
    d = DependencySet(bfs, fs)
    basisfunction_deps.append(d)

function_deps = []
for i in range(num_coefficients):
    # TODO: More dependencies depending on element
    bfs = (False,)*rank
    fs = tuple(i == j for j in range(num_coefficients))
    d = DependencySet(bfs, fs)
    function_deps.append(d)

(vinfo, code) = split_by_dependencies(integrand, formdata, basisfunction_deps, function_deps)

print "------ Final variable info:"
print vinfo

print "------ Stacks:"
for deps,stack in code.stacks.iteritems():
    print 
    print "Stack with deps =", deps
    print "\n".join(str(v) for v in stack)

print "------ Variable info:"
for count, v in code.variableinfo.iteritems():
    print
    print "v[%d] =" % count
    print v
print

#integrand = mark_dependencies(integrand)
#integrand = expand_compounds(integrand)#, skip=(Transpose, ...)
#integrand = compute_diffs(integrand)
#integrand = propagate_spatial_diffs(integrand)
##integrand = expand_compounds(integrand)
#(vinfo1, code1) = split_by_dependencies(integrand, formadata, basisfunction_deps, function_deps)
#integrand = mark_dependencies(integrand)
#(vinfo2, code2) = split_by_dependencies(integrand, formadata, basisfunction_deps, function_deps)

