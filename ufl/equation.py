# -*- coding: utf-8 -*-
"The Equation class, used to express equations like a == L."

# Copyright (C) 2012-2016 Anders Logg and Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.log import error
from ufl.utils.str import as_native_str
from ufl.utils.str import as_native_strings

# Export list for ufl.classes
__all_classes__ = as_native_strings(["Equation"])


class Equation(object):
    """This class is used to represent equations expressed by the "=="
    operator. Examples include a == L and F == 0 where a, L and F are
    Form objects."""

    def __init__(self, lhs, rhs):
        "Create equation lhs == rhs"
        self.lhs = lhs
        self.rhs = rhs

    def __bool__(self):
        """Evaluate bool(lhs_form == rhs_form).

        This will not trigger when setting 'equation = a == L',
        but when e.g. running 'if equation:'.
        """
        # NB!: pep8 will say you should use isinstance here, but we do
        #      actually want to compare the exact types in this case.
        # Not equal if types are not identical (i.e. not accepting
        # subclasses)
        if type(self.lhs) != type(self.rhs):  # noqa: E721
            return False
        # Try to delegate to equals function
        if hasattr(self.lhs, "equals"):
            return self.lhs.equals(self.rhs)
        elif hasattr(self.rhs, "equals"):
            return self.rhs.equals(self.lhs)
        else:
            error("Either lhs or rhs of Equation must implement self.equals(other).")
    __nonzero__ = __bool__

    def __eq__(self, other):
        "Compare two equations by comparing lhs and rhs."
        return isinstance(other, Equation) and \
            bool(self.lhs == other.lhs) and \
            bool(self.rhs == other.rhs)

    def __hash__(self):
        return hash((hash(self.lhs), hash(self.rhs)))

    def __repr__(self):
        r = "Equation(%s, %s)" % (repr(self.lhs), repr(self.rhs))
        return as_native_str(r)
