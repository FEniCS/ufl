"""This module contains algorithms for traversing expression trees in different ways."""

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
#
# Modified by Anders Logg, 2008

from ufl.log import error
from ufl.core.expr import Expr
from ufl.integral import Integral
from ufl.form import Form

#--- Traversal utilities ---

def iter_expressions(a):
    """Utility function to handle Form, Integral and any Expr
    the same way when inspecting expressions.
    Returns an iterable over Expr instances:
    - a is an Expr: (a,)
    - a is an Integral:  the integrand expression of a
    - a is a  Form:      all integrand expressions of all integrals
    """
    if isinstance(a, Form):
        return (itg.integrand() for itg in a.integrals())
    elif isinstance(a, Integral):
        return (a.integrand(),)
    elif isinstance(a, Expr):
        return (a,)
    error("Not an UFL type: %s" % str(type(a)))

# The rest is moved here:
#from ufl.core.traversal import pre_traversal, post_traversal, traverse_terminals, traverse_unique_terminals
