"""A collection of utility algorithms for handling UFL files."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-12-08"

from ufl.log import error, info
from ufl.form import Form
from ufl.function import Function
from ufl.algorithms.formdata import FormData
from ufl.algorithms.checks import validate_form

#--- Utilities to deal with form files ---

infostring = """An exception occured during evaluation of form file.
To find the location of the error, a temporary script
'%s' has been created and will now be executed:"""

def load_forms(filename):
    try:
        f = open(filename)
    except IOError:
        f = open(filename + ".ufl")
    
    # Read form file
    code = "from ufl import *\n"
    code += "\n".join(f.readlines())
    namespace = {}
    try:
        exec(code, namespace)
    except:
        tmpname = "ufl_analyse_tmp_form"
        tmpfile = tmpname + ".py"
        f = file(tmpfile, "w")
        f.write(code)
        f.close()
        info(infostring % tmpfile)
        m = __import__(tmpname)
        error("Aborting load_forms.")
    
    # Extract Form objects, and Function objects to get their names
    forms = []
    function_names = {}
    for k,v in namespace.iteritems():
        if isinstance(v, Form):
            forms.append((k,v))
        elif isinstance(v, Function):
            function_names[v] = k
    
    # Analyse validity of forms
    for k, v in forms:
        errors = validate_form(v)
        if errors:
            msg = "Found errors in form '%s':\n%s" % (k, errors)
            raise RuntimeError, msg
    
    # Construct FormData for each object
    formdatas = []
    for name, form in forms:
        fd = FormData(form, name)
        coefficient_names = [function_names.get(c, "w%d"%i) for (i,c) in enumerate(fd.coefficients)]
        fd.coefficient_names = coefficient_names
        formdatas.append(fd)
    
    return formdatas

