"""A collection of utility algorithms for inspection,
conversion or transformation of UFL objects."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-05-20"

from itertools import chain

from output import *
from base import *
from traversal import *
from analysis import *
from transformations import *


#--- Utilities to deal with form files ---

def load_forms(filename):
    # Read form file
    code = "from ufl import *\n"
    code += "\n".join(file(filename).readlines())
    namespace = {}
    try:
        exec(code, namespace)
    except:
        tmpname = "ufl_analyse_tmp_form"
        tmpfile = tmpname + ".py"
        f = file(tmpfile, "w")
        f.write(code)
        f.close()
        ufl_info("""\
An exception occured during evaluation of form file.
To find the location of the error, a temporary script
'%s' has been created and will now be run:""" % tmpfile)
        m = __import__(tmpname)
        ufl_error("Aborting load_forms.")
    
    # Extract Form objects
    forms = []
    for k,v in namespace.iteritems():
        if isinstance(v, Form):
            forms.append((k,v))
    
    return forms


#--- Utilities for constructing informative strings from UFL objects ---

def integral_info(integral):
    s  = "  Integral over %s domain %d:\n" % (integral._domain_type, integral._domain_id)
    s += "    Integrand expression representation:\n"
    s += "      %s\n" % repr(integral._integrand)
    s += "    Integrand expression short form:\n"
    s += "      %s" % str(integral._integrand)
    return s

def form_info(form):
    ufl_assert(isinstance(form, Form), "Expecting a Form.")
    
    bf = basisfunctions(form)
    cf = coefficients(form)
    
    ci = form.cell_integrals()
    ei = form.exterior_facet_integrals()
    ii = form.interior_facet_integrals()
    
    s  = "Form info:\n"
    s += "  rank:                          %d\n" % len(bf)
    s += "  num_coefficients:              %d\n" % len(cf)
    s += "  num_cell_integrals:            %d\n" % len(ci)
    s += "  num_exterior_facet_integrals:  %d\n" % len(ei)
    s += "  num_interior_facet_integrals:  %d\n" % len(ii)
    
    for f in cf:
        if f._name:
            s += "\n"
            s += "  Function %d is named '%s'" % (f._count, f._name)
    s += "\n"
    
    for itg in ci:
        s += "\n"
        s += integral_info(itg)
    for itg in ei:
        s += "\n"
        s += integral_info(itg)
    for itg in ii:
        s += "\n"
        s += integral_info(itg)
    return s


