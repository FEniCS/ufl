# Copyright (C) 2010-2013 Marie E. Rognes
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

from ufl.assertions import ufl_assert

def FEEC_aliases(name, cell, r, k):
    """

    FEEC_aliases(name, cell, r, k):

    name:                 "P Lambda" or "P- Lambda"
    cell:                 "interval", "triangle", "tetrahedron"
    r (polynomial degree): 1 <= r < ...
    k (form degree):       0 <= k <= n

    where n is the topological dimension of the cell.

    The families

    P_r Lambda^k
    P-_r Lambda^k

    map to H^1/H(curl)/H(div)/L^2 conforming finite element spaces
    based on the notation used in"Finite element exterior calculus, homological
    techniques and applications,", Arnold, Falk and Winther, Acta
    Numerica, 2006, Table 5.1 and 5.2 (p. 60)
    """

    ufl_assert(cell is not None, "Cannot get dimension from undefined cell.")
    tdim = cell.topological_dimension()
    ufl_assert(k in set(range(0, tdim+1)),\
               "k-forms only defined for k in [0, n]")

    if k == 0:
        family = "CG"
    elif k == tdim:
        family = "DG"
        if name == "P- Lambda":
            r = r - 1
    elif k == 1:
        if name == "P Lambda":
            family = "Nedelec 2nd kind H(curl)"
        elif name == "P- Lambda":
            family = "Nedelec 1st kind H(curl)"
    elif k == tdim - 1:
        if name == "P Lambda":
            family = "Nedelec 2nd kind H(div)"
        elif name == "P- Lambda":
            family = "Nedelec 1st kind H(div)"

    return (family, cell, r)

