"""A collection of utility algorithms for handling UFL files."""

from __future__ import with_statement

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2009-02-12"

import os
import time
from ufl.log import error, info
from ufl.form import Form
from ufl.function import Function
from ufl.algorithms.formdata import FormData
from ufl.algorithms.checks import validate_form

#--- Utilities to deal with form files ---

infostring = """An exception occured during evaluation of form file.
To help you find the location of the error, a temporary script
'%s'
has been created and will now be executed with debug output enabled:"""

def load_forms(filename):
    if not os.path.exists(filename):
        filename = filename + ".ufl"
    if not os.path.exists(filename):
        error("File '%s' doesn't exists." % filename)
    
    # Read form file and prepend import
    with open(filename) as f:
        fcode = f.read()
        code = "from ufl import *\n" + fcode
    
    # Execute code
    namespace = {}
    try:
        exec code in namespace
    except:
        # Dump code for debugging if this fails
        basename = os.path.splitext(os.path.basename(filename))[0]
        basename = "%s_debug" % basename
        pyname = "%s.py" % basename
        code = "#!/usr/bin/env python\nfrom ufl import *\nset_level(DEBUG)\n" + fcode
        with file(pyname, "w") as f:
            f.write(code)
        info(infostring % pyname)
        m = __import__(basename)
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
            error("Found errors in form '%s':\n%s" % (k, errors))
    
    # Construct FormData for each object
    formdatas = []
    for name, form in forms:
        fd = FormData(form, name)
        for (i,c) in enumerate(fd.coefficients):
            orig_c = fd.original_arguments[c]
            fd.coefficient_names[i] = function_names.get(orig_c, "w%d"%i)
        formdatas.append(fd)
    
    return formdatas

