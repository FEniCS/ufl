# -*- coding: utf-8 -*-
"""A collection of utility algorithms for handling UFL files."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2008-2009.
# Modified by Marie E. Rognes, 2011.

import io
import os
import re

from ufl.argument import Argument
from ufl.coefficient import Coefficient
from ufl.constant import Constant
from ufl.core.expr import Expr
from ufl.finiteelement import FiniteElementBase
from ufl.form import Form
from ufl.utils.sorting import sorted_by_key


class FileData(object):
    def __init__(self):
        self.elements = []
        self.coefficients = []
        self.expressions = []
        self.forms = []
        self.object_names = {}
        self.object_by_name = {}
        self.reserved_objects = {}

    def __bool__(self):
        return bool(self.elements or self.coefficients or self.forms or self.expressions or  # noqa: W504
                    self.object_names or self.object_by_name or self.reserved_objects)

    __nonzero__ = __bool__


def read_lines_decoded(fn):
    r = re.compile(b".*coding: *([^ ]+)")

    def match(line):
        return r.match(line, re.ASCII)

    # First read lines as bytes
    with io.open(fn, "rb") as f:
        lines = f.readlines()

    # Check for coding: in the first two lines
    for i in range(min(2, len(lines))):
        m = match(lines[i])
        if m:
            encoding, = m.groups()
            # Drop encoding line
            lines = lines[:i] + lines[i + 1:]
            break
    else:
        # Default to utf-8 (works for ascii files
        # as well, default for python files in py3)
        encoding = "utf-8"

    # Decode all lines
    lines = [line.decode(encoding=encoding) for line in lines]
    return lines


def read_ufl_file(filename):
    "Read a UFL file."
    if not os.path.exists(filename):
        raise ValueError(f"File '{filename}' doesn't exist.")
    lines = read_lines_decoded(filename)
    code = "".join(lines)
    return code


def execute_ufl_code(uflcode):
    # Execute code
    namespace = {}
    exec(uflcode, namespace)
    return namespace


def interpret_ufl_namespace(namespace):
    "Takes a namespace dict from an executed ufl file and converts it to a FileData object."
    # Object to hold all returned data
    ufd = FileData()

    # Extract object names for Form, Coefficient and FiniteElementBase objects
    # The use of id(obj) as key in object_names is necessary
    # because we need to distinguish between instances,
    # and not just between objects with different values.
    for name, value in sorted_by_key(namespace):
        # Store objects by reserved name OR instance id
        reserved_names = ("unknown",)  # Currently only one reserved name
        if name in reserved_names:
            # Store objects with reserved names
            ufd.reserved_objects[name] = value
            # FIXME: Remove after FFC is updated to use reserved_objects:
            ufd.object_names[name] = value
            ufd.object_by_name[name] = value
        elif isinstance(value, (FiniteElementBase, Coefficient, Constant, Argument, Form, Expr)):
            # Store instance <-> name mappings for important objects
            # without a reserved name
            ufd.object_names[id(value)] = name
            ufd.object_by_name[name] = value

    # Get list of exported forms
    forms = namespace.get("forms")
    if forms is None:
        # Get forms from object_by_name, which has already mapped
        # tuple->Form where needed
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
    if not isinstance(ufd.forms, (list, tuple)):
        raise ValueError(f"Expecting 'forms' to be a list or tuple, not '{type(ufd.forms)}'.")
    if not all(isinstance(a, Form) for a in ufd.forms):
        raise ValueError("Expecting 'forms' to be a list of Form instances.")

    # Get list of exported elements
    elements = namespace.get("elements")
    if elements is None:
        elements = [ufd.object_by_name.get(name) for name in ("element",)]
        elements = [e for e in elements if e is not None]
    ufd.elements = elements

    # Validate types
    if not isinstance(ufd.elements, (list, tuple)):
        raise ValueError(f"Expecting 'elements' to be a list or tuple, not '{type(ufd.elements)}''.")
    if not all(isinstance(e, FiniteElementBase) for e in ufd.elements):
        raise ValueError("Expecting 'elements' to be a list of FiniteElementBase instances.")

    # Get list of exported coefficients
    functions = []
    ufd.coefficients = namespace.get("coefficients", functions)

    # Validate types
    if not isinstance(ufd.coefficients, (list, tuple)):
        raise ValueError(f"Expecting 'coefficients' to be a list or tuple, not '{type(ufd.coefficients)}'.")
    if not all(isinstance(e, Coefficient) for e in ufd.coefficients):
        raise ValueError("Expecting 'coefficients' to be a list of Coefficient instances.")

    # Get list of exported expressions
    ufd.expressions = namespace.get("expressions", [])

    # Validate types
    if not isinstance(ufd.expressions, (list, tuple)):
        raise ValueError(f"Expecting 'expressions' to be a list or tuple, not '{type(ufd.expressions)}'.")
    if not all(isinstance(e[0], Expr) for e in ufd.expressions):
        raise ValueError("Expecting 'expressions' to be a list of Expr instances.")

    # Return file data
    return ufd


def load_ufl_file(filename):
    "Load a UFL file with elements, coefficients, expressions and forms."
    # Read code from file and execute it
    uflcode = read_ufl_file(filename)
    namespace = execute_ufl_code(uflcode)
    return interpret_ufl_namespace(namespace)


def load_forms(filename):
    "Return a list of all forms in a file."
    ufd = load_ufl_file(filename)
    return ufd.forms
