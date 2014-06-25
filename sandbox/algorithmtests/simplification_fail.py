
#
# An example that improved simplification could optimize.
#

from ufl import *
from ufl.algorithms import expand_indices

x = triangle.x

e = (x[i] - x[i])*x[i]

a = e*dx

print(a)
#{ sum_{i_0} ((x)[i_0]) * (((x)[i_0]) + -1 * ((x)[i_0]))  } * dx0

fd = a.form_data()
b = fd.form

print(b)
#{ sum_{i_0} ((x)[i_0]) * (((x)[i_0]) + -1 * ((x)[i_0]))  } * dx0

print(expand_indices(b))
#{ ((x)[0]) * (((x)[0]) + -1 * ((x)[0])) + ((x)[1]) * (((x)[1]) + -1 * ((x)[1])) } * dx0

