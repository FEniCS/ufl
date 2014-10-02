"Representation of the reference value of a function."

# Copyright (C) 2008-2014 Martin Sandve Alnes
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

from ufl.core.ufl_type import ufl_type
from ufl.core.operator import Operator

@ufl_type(num_ops=1, is_index_free=True)
class ReferenceValue(Operator):
    "Representation of the reference cell value of a form argument."
    __slots__ = ()

    def __init__(self, f):
        Operator.__init__(self, (f,))

    @property
    def ufl_shape(self):
        return self.ufl_operands[0].element().reference_value_shape()

    def reconstruct(self, op):
        "Return a new object of the same type with new operands."
        return self._ufl_class_(op)

    def evaluate(self, x, mapping, component, index_values, derivatives=()):
        "Get child from mapping and return the component asked for."
        error("Evaluate not implemented.")

    @property
    def ufl_shape(self):
        return self.ufl_operands[0].ufl_shape + (self._dim,)

    def __str__(self):
        return "reference_value(%s)" % self.ufl_operands[0]

    def __repr__(self):
        return "ReferenceValue(%r)" % self.ufl_operands[0]
