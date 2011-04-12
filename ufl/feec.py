
__authors__ = "Marie E. Rognes"
__copyright__ = "Copyright (C) 2010-2011 Marie E. Rognes"
__license__  = "GNU LGPL version 3 or any later version"

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

    ufl_assert(k in set(range(0, cell.topological_dimension()+1)),
               "k-forms only defined for k in [0, n]")

    if k == 0:
        family = "CG"
    elif k == cell.topological_dimension():
        family = "DG"
        if name == "P- Lambda":
            r = r - 1
    elif k == 1:
        if name == "P Lambda":
            family = "Nedelec 2nd kind H(curl)"
        elif name == "P- Lambda":
            family = "Nedelec 1st kind H(curl)"
    elif k == cell.topological_dimension() - 1:
        if name == "P Lambda":
            family = "Nedelec 2nd kind H(div)"
        elif name == "P- Lambda":
            family = "Nedelec 1st kind H(div)"

    return (family, cell, r)

