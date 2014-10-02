"""Functions to check properties of forms and integrals."""

# Copyright (C) 2008-2014 Martin Sandve Alnes and Anders Logg
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

from ufl.log import warning, debug
from ufl.algorithms.traversal import iter_expressions
from ufl.algorithms.argument_dependencies import extract_argument_dependencies, NotMultiLinearException

#--- Utilities for checking properties of forms ---

def is_multilinear(form):
    "Check if form is multilinear in arguments."
    # An attempt at implementing is_multilinear using extract_argument_dependencies.
    # TODO: This has some false negatives for "multiple configurations". (Does it still? Needs testing!)
    # TODO: FFC probably needs a variant of this which checks for some sorts of linearity
    #       in Coefficients as well, this should be a fairly simple extension of the current algorithm.
    try:
        for e in iter_expressions(form):
            deps = extract_argument_dependencies(e)
            nargs = [len(d) for d in deps]
            if len(nargs) == 0:
                debug("This form is a functional.")
            if len(nargs) == 1:
                debug("This form is linear in %d arguments." % nargs[0])
            if len(nargs) > 1:
                warning("This form has more than one argument "\
                    "'configuration', it has terms that are linear in %s "\
                    "arguments respectively." % str(nargs))

    except NotMultiLinearException as msg:
        warning("Form is not multilinear, the offending term is: %s" % msg)
        return False

    return True
