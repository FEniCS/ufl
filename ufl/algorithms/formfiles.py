"""A collection of utility algorithms for handling UFL files."""

from __future__ import with_statement

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2009-03-31"

import os
import time

from ufl.log import error, info
from ufl.assertions import ufl_assert
from ufl.form import Form
from ufl.finiteelement import FiniteElementBase
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
    all_forms = []
    form_names = {}
    function_names = {}
    basis_function_names = {}
    for name, value in namespace.iteritems():
        if isinstance(value, (Form, tuple)):
            all_forms.append(value)
            form_names[id(value)] = name
        elif isinstance(value, Function):
            function_names[id(value)] = name
        elif isinstance(value, BasisFunction):
            basis_function_names[id(value)] = name

    # Get list of forms
    forms = namespace.get("forms")
    if forms is None:
        forms = [namespace.get(name) for name in ("a", "L", "M")]
        forms = [a for a in forms if a is not None]
    # Convert tuple type forms to Form instances
    ufl_assert(isinstance(forms, (list, tuple)),
        "Expecting 'forms' to be a list or tuple, not '%s'." % type(forms))
    ufl_assert(all(isinstance(a, Form) for a in forms),
        "Expecting 'forms' to be a list of Form instances.")
    
    # Get list of elements
    elements = namespace.get("elements")
    if elements is None:
        elements = [namespace.get(name) for name in ("element",)]
        elements = [e for e in elements if not e is None]
    ufl_assert(isinstance(elements,  (list, tuple)),
        "Expecting 'elements' to be a list or tuple, not '%s'." % type(elements))
    ufl_assert(all(isinstance(e, FiniteElementBase) for e in elements),
        "Expecting 'elements' to be a list of FiniteElementBase instances.")

    # TODO: Get a list of functions as well?
    # Relevant functions will be extracted from forms
    functions = namespace.get("functions", [])
    if functions:
        warning("List of functions not implemented.")
    ufl_assert(isinstance(functions, (list, tuple)),
        "Expecting 'functions' to be a list or tuple, not '%s'." % type(functions))
    ufl_assert(all(isinstance(e, Function) for e in functions),
        "Expecting 'functions' to be a list of Function instances.")

    # Analyse validity of forms
    for form in forms:
        validate_form(form)
        #errors = validate_form(form) # TODO: validate_form raises exception, it doesn't return errors
        #if errors:
        #    error("Found errors in form '%s':\n%s" % (form_names[id(form)], errors))

    # Construct FormData for each object and attach names
    formdatas = []
    for form in forms:
        # Using form_data() ensures FormData is only constructed once
        fd = form.form_data()
        fd.name = form_names[id(form)]
        for (i, f) in enumerate(fd.original_functions):
            fd.function_names[i] = function_names.get(id(f), "w%d"%i)
        for (i, f) in enumerate(fd.original_basis_functions):
            fd.basis_function_names[i] = basis_function_names.get(id(f), "w%d"%i)
        formdatas.append(fd)

    # FIXME: Return elements as well (postphoned since it will break other code)
    #return elements, forms
    #return elements, functions, forms
    return forms

