"""Algorithm sketch to build canonical data structure for integrals over subdomains."""

from ufl import *

# Transitional helper constructor
from ufl.integral import Integral2

from ufl.algorithms.domain_analysis import (extract_subdomain_data_from_integral_dict,
                                            extract_integral_data_from_integral_dict)

# Run for testing and inspection
def test():

    # Mock objects for compiler data and solver data
    comp1 = [1, 2, 3]
    comp2 = ('a', 'b')
    comp3 = {'1':1}
    sol1 = (0, 3, 5)
    sol2 = (0, 3, 7)

    # Basic UFL expressions for integrands
    V = FiniteElement("CG", triangle, 1)
    f = Coefficient(V)
    g = Coefficient(V)
    h = Coefficient(V)

    # Mock list of Integral objects (Integral2 is a dummy factory function for better constructor signature)
    integrals = {}
    integrals["cell"] = [# Integrals over 0 with no compiler_data:
                         Integral2(f, "cell", 0, None, None),
                         Integral2(g, "cell", 0, None, sol1),
                         # Integrals over 0 with different compiler_data:
                         Integral2(f**2, "cell", 0, comp1, None),
                         Integral2(g**2, "cell", 0, comp2, None),
                         # Integrals over 1 with same compiler_data object:
                         Integral2(f**3, "cell", 1, comp1, None),
                         Integral2(g**3, "cell", 1, comp1, sol1),
                         # Integral over 0 and 1 with compiler_data object found in 0 but not 1 above:
                         Integral2(f**4, "cell", (0, 1), comp2, None),
                         # Integral over 0 and 1 with its own compiler_data object:
                         Integral2(g**4, "cell", (0, 1), comp3, None),
                         # Integral over 0 and 1 no compiler_data object:
                         Integral2(h/3, "cell", (0, 1), None, None),
                         # Integral over everywhere with no compiler data:
                         Integral2(h/2, "cell", Measure.DOMAIN_ID_EVERYWHERE, None, None),
                         ]

    # Create form from all mock integrals to make test more realistic
    form = Form(integrals["cell"])

    subdomain_data = extract_subdomain_data_from_integral_dict(form._dintegrals)
    integral_data = extract_integral_data_from_integral_dict(form._dintegrals)

    print()
    print("Domain data:")
    print(subdomain_data)
    print()

    print()
    print("Integral data:")
    for ida in integral_data:
        print(ida)
    print()

test()
