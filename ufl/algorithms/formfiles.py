"""A collection of utility algorithms for handling UFL files."""

from __future__ import with_statement

# Copyright (C) 2008-2013 Martin Sandve Alnes
#
# This file is part of UFL.
#
# UFL is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# UFL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UFL. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg, 2008-2009.
# Modified by Marie E. Rognes, 2011.
#
# First added:  2008-03-14
# Last changed: 2012-12-11

import os, re
from ufl.log import error, warning
from ufl.common import sorted_items
from ufl.assertions import ufl_assert
from ufl.form import Form
from ufl.finiteelement import FiniteElementBase
from ufl.expr import Expr
from ufl.argument import Argument
from ufl.coefficient import Coefficient

class FileData(object):
    def __init__(self):
        self.elements     = []
        self.coefficients = []
        self.expressions  = []
        self.forms        = []
        self.object_names = {}
        self.object_by_name = {}
        self.reserved_objects = {}

    def __nonzero__(self):
        return bool(self.elements or self.coefficients or self.forms or self.expressions or\
                    self.object_names or self.object_by_name or self.reserved_objects)

infostring = """An exception occured during evaluation of form file.
To help you find the location of the error, a temporary script
'%s'
has been created and will now be executed with debug output enabled:"""

def replace_include_statements(code):
    "Replace '#include foo.ufl' statements with contents of foo.ufl."
    if "#include" in code:
        lines = code.split("\n")
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
        return "\n".join(l.rstrip() for l in newlines)
    return code

def read_ufl_file(filename):
    "Read a .ufl file, handling file extension, file existance, and #include replacement."
    if not os.path.exists(filename) and filename[-4:] != ".ufl":
        filename = filename + ".ufl"
    if not os.path.exists(filename):
        error("File '%s' doesn't exist." % filename)
    with open(filename) as f:
        code = replace_include_statements(f.read())
    return code

def execute_ufl_code(uflcode, filename):
    # Execute code
    namespace = {}
    try:
        pycode = "from ufl import *\n" + uflcode
        exec pycode in namespace
    except:
        # Dump python code for debugging if this fails
        basename = os.path.splitext(os.path.basename(filename))[0]
        basename = "%s_debug" % basename
        pyname = "%s.py" % basename
        pycode = "#!/usr/bin/env python\nfrom ufl import *\nset_level(DEBUG)\n" + uflcode
        with file(pyname, "w") as f:
            f.write(pycode)
        warning(infostring % pyname)
        m = __import__(basename)
        error("An error occured, aborting load_forms.")
    return namespace

def interpret_ufl_namespace(namespace):
    "Takes a namespace dict from an executed ufl file and converts it to a FileData object."
    # Object to hold all returned data
    ufd = FileData()

    # Extract object names for Form, Coefficient and FiniteElementBase objects
    # The use of id(obj) as key in object_names is necessary
    # because we need to distinguish between instances,
    # and not just between objects with different values.
    for name, value in sorted_items(namespace):
        # Store objects by reserved name OR instance id
        reserved_names = ("unknown",) # Currently only one reserved name
        if name in reserved_names:
            # Store objects with reserved names
            ufd.reserved_objects[name] = value
            # FIXME: Remove after FFC is updated to use reserved_objects:
            ufd.object_names[name] = value
            ufd.object_by_name[name] = value
        elif isinstance(value, (FiniteElementBase, Coefficient, Argument, Form, Expr)):
            # Store instance <-> name mappings for important objects without a reserved name
            ufd.object_names[id(value)] = name
            ufd.object_by_name[name] = value

    # Get list of exported forms
    forms = namespace.get("forms")
    if forms is None:
        # Get forms from object_by_name, which has already mapped tuple->Form where needed
        def get_form(name):
            form = ufd.object_by_name.get(name)
            return form if isinstance(form, Form) else None
        a_form = get_form("a")
        L_form = get_form("L")
        M_form = get_form("M")
        forms = [a_form, L_form, M_form]
        # Add forms F and J if not "a" and "L" are used
        if a_form is None or L_form is None:
            F_form = get_form("F")
            J_form = get_form("J")
            forms += [F_form, J_form]
        # Remove Nones
        forms = [f for f in forms if isinstance(f, Form)]
    ufd.forms = forms

    # Validate types
    ufl_assert(isinstance(ufd.forms, (list, tuple)),
        "Expecting 'forms' to be a list or tuple, not '%s'." % type(ufd.forms))
    ufl_assert(all(isinstance(a, Form) for a in ufd.forms),
        "Expecting 'forms' to be a list of Form instances.")

    # Get list of exported elements
    elements = namespace.get("elements")
    if elements is None:
        elements = [ufd.object_by_name.get(name) for name in ("element",)]
        elements = [e for e in elements if e is not None]
    ufd.elements = elements

    # Validate types
    ufl_assert(isinstance(ufd.elements,  (list, tuple)),
        "Expecting 'elements' to be a list or tuple, not '%s'." % type(ufd.elements))
    ufl_assert(all(isinstance(e, FiniteElementBase) for e in ufd.elements),
        "Expecting 'elements' to be a list of FiniteElementBase instances.")

    # Get list of exported coefficients
    # TODO: Temporarily letting 'coefficients' override 'functions', but allow 'functions' for compatibility
    functions = namespace.get("functions", [])
    if functions:
        warning("Deprecation warning: Rename 'functions' to 'coefficients' to export coefficients.")
    ufd.coefficients = namespace.get("coefficients", functions)
    #ufd.coefficients = namespace.get("coefficients", [])

    # Validate types
    ufl_assert(isinstance(ufd.coefficients, (list, tuple)),
        "Expecting 'coefficients' to be a list or tuple, not '%s'." % type(ufd.coefficients))
    ufl_assert(all(isinstance(e, Coefficient) for e in ufd.coefficients),
        "Expecting 'coefficients' to be a list of Coefficient instances.")

    # Get list of exported expressions
    ufd.expressions = namespace.get("expressions", [])

    # Validate types
    ufl_assert(isinstance(ufd.expressions, (list, tuple)),
        "Expecting 'expressions' to be a list or tuple, not '%s'." % type(ufd.expressions))
    ufl_assert(all(isinstance(e, Expr) for e in ufd.expressions),
        "Expecting 'expressions' to be a list of Expr instances.")

    # Return file data
    return ufd

def load_ufl_file(filename):
    "Load a .ufl file with elements, coefficients and forms."
    # Read code from file and execute it
    uflcode = read_ufl_file(filename)
    namespace = execute_ufl_code(uflcode, filename)
    return interpret_ufl_namespace(namespace)

def load_forms(filename):
    "Return a list of all forms in a file."
    ufd = load_ufl_file(filename)
    return ufd.forms
