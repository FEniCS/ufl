__authors__ = "Marie E. Rognes (meg@simula.no)"
__copyright__ = "Copyright (C) 2011 " + __authors__
__license__  = "GNU LGPL version 3 or any later version"
__date__ = "2011-01-17 -- 2011-01-17"

from ufl.finiteelement import FiniteElement, MixedElement

def change_regularity(element, family):
    """
    For a given finite element, return the corresponding space
    specified by 'family'.
    """

    n = element.num_sub_elements()
    if n > 0:
        subs = element.sub_elements()
        return MixedElement([change_regularity(subs[i], family)
                             for i in range(n)])
    shape = element.value_shape()
    if not shape:
        return FiniteElement(family, element.cell(), element.degree())

    return MixedElement([FiniteElement(family, element.cell(), element.degree())
                               for i in range(shape[0])])

def tear(V):
    "For a finite element, return the corresponding discontinuous element."
    W = change_regularity(V, "DG")
    return W

def increase_order(element):
    "Return element of same family, but a polynomial degree higher."

    n = element.num_sub_elements()
    if n > 0:
        subs = element.sub_elements()
        return MixedElement([increase_order(subs[i]) for i in range(n)])

    if element.family() == "Real":
        return element

    return FiniteElement(element.family(), element.cell(), element.degree()+1)
