"Precedence handling."

# Copyright (C) 2008-2011 Martin Sandve Alnes
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
# First added:  2009-03-27
# Last changed: 2009-04-19

from ufl.log import error, warning
from ufl.assertions import ufl_assert
from ufl.common import product, mergedicts, subdict
from ufl.expr import Expr
from ufl.terminal import Terminal

def parstr(child, parent, pre="(", post=")"):
    s = str(child)
    # We want child to be evaluated fully first,
    # so if the parent has higher precedence
    # we wrap in ().

    # Operators where operands are always parenthesized 
    if parent._precedence == 0:
        return pre + s + post

    # If parent operator binds stronger than child, must parenthesize child
    if parent._precedence > child._precedence:
        return pre + s + post

    # Nothing needed
    return s
