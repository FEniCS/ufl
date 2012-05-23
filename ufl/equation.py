"The Equation class, used to express equations like a == L."

# Copyright (C) 2012 Anders Logg and Martin Sandve Alnes
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
# First added:  2011-06-21
# Last changed: 2011-06-22

class Equation:

    """This class is used to represent equations expressed by the "=="
    operator. Examples include a == L and F == 0 where a, L and F are
    Form objects."""

    def __init__(self, lhs, rhs):
        "Create equation lhs == rhs"
        self.lhs = lhs
        self.rhs = rhs

    def __nonzero__(self):
        return type(self.lhs) == type(self.rhs) and \
            repr(self.lhs) == repr(self.rhs) # REPR not a problem

    def __eq__(self, other):
        return isinstance(other, Equation) and \
            bool(self.lhs == other.lhs) and \
            bool(self.rhs == other.rhs)

    def __repr__(self):
        return "Equation(%r, %r)" % (self.lhs, self.rhs)
