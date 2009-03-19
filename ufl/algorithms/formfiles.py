"""A collection of utility algorithms for handling UFL files."""

from __future__ import with_statement

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2009-03-05"

import os
import time
from ufl.log import error, info
from ufl.form import Form
from ufl.function import Function
from ufl.basisfunction import BasisFunction
from ufl.algorithms.formdata import FormData
from ufl.algorithms.checks import validate_form

#--- Utilities to deal with form files ---

infostring = """An exception occured during evaluation of form file.
To help you find the location of the error, a temporary script
'%s'
has been created and will now be executed with debug output enabled:"""

def load_forms(filename):
    if not os.path.exists(filename) and filename[-4:] != ".ufl":
        filename = filename + ".ufl"
    if not os.path.exists(filename):
        error("File '%s' doesn't exist." % filename)
    
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
    form_names = []
    function_names = {}
    basis_function_names = {}
    for name, value in namespace.iteritems():
        if isinstance(value, Form):
            forms.append(value)
            form_names.append(name)
        elif isinstance(value, Function):
            function_names[value] = name
        elif isinstance(value, BasisFunction):
            basis_function_names[value] = name
    
    # Analyse validity of forms
    for k, v in zip(form_names, forms):
        validate_form(v)
        #errors = validate_form(v) # TODO: validate_form raises exception, it doesn't return errors
        #if errors:
        #    error("Found errors in form '%s':\n%s" % (k, errors))
    
    # Construct FormData for each object
    formdatas = []
    for name, form in zip(form_names, forms):
        # Using form_data() ensures FormData is only constructed once
        fd = form.form_data()
        fd.name = name
        for (i, f) in enumerate(fd.original_functions):
            fd.function_names[i] = function_names.get(f, "w%d"%i)
        for (i, f) in enumerate(fd.original_basis_functions):
            fd.basis_function_names[i] = basis_function_names.get(f, "w%d"%i)
        formdatas.append(fd)
    
    return forms

