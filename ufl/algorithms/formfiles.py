"""A collection of utility algorithms for handling UFL files."""

from __future__ import with_statement

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14"

# Modified by Anders Logg, 2008-2009.
# Modified by Marie E. Rognes, 2011.

# Last changed: 2011-01-08

import os
import time
import re

from ufl.log import error, warning, info
from ufl.assertions import ufl_assert
from ufl.form import Form
from ufl.finiteelement import FiniteElementBase
from ufl.argument import Argument
from ufl.coefficient import Coefficient
from ufl.algorithms.formdata import FormData
from ufl.algorithms.checks import validate_form
from ufl.algorithms.tuplenotation import as_form

class FileData(object):
    def __init__(self):
        self.elements     = []
        self.functions    = []
        self.forms        = []
        self.object_names = {}

    def __nonzero__(self):
        return bool(self.elements or self.functions or self.forms)

infostring = """An exception occured during evaluation of form file.
To help you find the location of the error, a temporary script
'%s'
has been created and will now be executed with debug output enabled:"""

def read_ufl_file(filename):
    "Load a .ufl file with elements, functions and forms."
    if not os.path.exists(filename) and filename[-4:] != ".ufl":
        filename = filename + ".ufl"
    if not os.path.exists(filename):
        error("File '%s' doesn't exist." % filename)

    # Read form file and prepend import
    with open(filename) as f:
        fcode = f.read()
        # Replace #include foo.ufl statements with contents of foo.ufl
        if "#include" in fcode:
            lines = fcode.split("\n")
            newlines = []
            regexp = re.compile(r"^#include (.*)$")
            for l in lines:
                m = regexp.search(l)
                if m:
                    fn = m.groups()[0]
                    newlines.append("# --- begin %s" % fn)
                    newlines.extend(open(fn).readlines())
                    newlines.append("# --- end %s" % fn)
                else:
                    newlines.append(l)
            fcode = "\n".join(l.rstrip() for l in newlines)

    return fcode

def load_ufl_file(filename):
    "Load a .ufl file with elements, functions and forms."
    if not os.path.exists(filename) and filename[-4:] != ".ufl":
        filename = filename + ".ufl"
    if not os.path.exists(filename):
        error("File '%s' doesn't exist." % filename)

    # Read code
    fcode = read_ufl_file(filename)

    # Execute code
    namespace = {}
    try:
        code = "from ufl import *\n" + fcode
        exec code in namespace
    except:
        # Dump code for debugging if this fails
        basename = os.path.splitext(os.path.basename(filename))[0]
        basename = "%s_debug" % basename
        pyname = "%s.py" % basename
        code = "#!/usr/bin/env python\nfrom ufl import *\nset_level(DEBUG)\n" + fcode
        with file(pyname, "w") as f:
            f.write(code)
        warning(infostring % pyname)
        m = __import__(basename)
        error("An error occured, aborting load_forms.")

    # Object to hold all returned data
    ufd = FileData()

    # Extract Form objects, and Coefficient objects to get their names
    all_forms = []

    # Is id() necessary? Yes! Most likely. Otherwise some __hash__ thing
    # might kick in and give us the same hash for different objects

    # Extract object names
    for name, value in namespace.iteritems():
        if isinstance(value, (Form, tuple)):
            # Convert tuple notation to form
            all_forms.append(as_form(value))
            ufd.object_names[id(as_form(value))] = name
        elif isinstance(value, (Coefficient, Argument)):
            if name is "unknown":
                ufd.object_names[name] = value
            ufd.object_names[id(value)] = name

    # Get list of forms
    forms = namespace.get("forms")
    if forms is None:
        forms = [namespace.get(name) for name in ("a", "L", "M")]
        # Convert tuple notation to form
        forms = [as_form(a) for a in forms if a is not None]
    # Convert tuple type forms to Form instances
    ufl_assert(isinstance(forms, (list, tuple)),
        "Expecting 'forms' to be a list or tuple, not '%s'." % type(forms))
    ufl_assert(all(isinstance(a, Form) for a in forms),
        "Expecting 'forms' to be a list of Form instances.")
    ufd.forms = forms

    # Get list of elements
    elements = namespace.get("elements")
    if elements is None:
        elements = [namespace.get(name) for name in ("element",)]
        elements = [e for e in elements if not e is None]
    ufl_assert(isinstance(elements,  (list, tuple)),
        "Expecting 'elements' to be a list or tuple, not '%s'." % type(elements))
    ufl_assert(all(isinstance(e, FiniteElementBase) for e in elements),
        "Expecting 'elements' to be a list of FiniteElementBase instances.")
    ufd.elements = elements

    # Relevant functions will be extracted from forms
    functions = namespace.get("functions", [])
    if functions:
        warning("List of functions not implemented.")
    ufl_assert(isinstance(functions, (list, tuple)),
        "Expecting 'functions' to be a list or tuple, not '%s'." % type(functions))
    ufl_assert(all(isinstance(e, Coefficient) for e in functions),
        "Expecting 'functions' to be a list of Coefficient instances.")
    ufd.functions = functions

    # FIXME: Need to call validate_form elsewhere!

    # Analyse validity of forms
    #for form in forms:
    #    validate_form(form)
        #errors = validate_form(form) # TODO: validate_form raises exception, it doesn't return errors
        #if errors:
        #    error("Found errors in form '%s':\n%s" % (form_names[id(form)], errors))

    # Return file data
    return ufd

def load_forms(filename):
    "Return a list of all forms in a file."
    ufd = load_ufl_file(filename)
    return ufd.forms
