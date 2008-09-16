"""A collection of utility algorithms for handling UFL files."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-09-16"

from ..output import ufl_error, ufl_info
from ..form import Form

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

