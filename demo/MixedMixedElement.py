# Copyright (C) 2007 Anders Logg
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
# A mixed element of mixed elements
from ufl import triangle
from ufl.finiteelement import FiniteElement, MixedElement
from ufl.pull_back import identity_pull_back
from ufl.sobolevspace import H1

P3 = FiniteElement("Lagrange", triangle, 3, (), (), identity_pull_back, H1)

element = MixedElement([[P3, P3], [P3, P3]])
