
from ufl import *
from ufl.base import as_ufl

a = as_ufl(pi)
print "sin(pi) =", sin(a)
print "cos(pi) =", cos(a)
print "exp(pi), pi =", exp(a), ln(exp(a))
print "ln(pi), pi =", ln(a), exp(ln(a))
print "sqrt(pi), pi =", sqrt(a), sqrt(a)**2

