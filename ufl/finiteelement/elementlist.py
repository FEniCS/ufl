"""This module provides an extensive list of predefined finite element
families. Users or more likely, form compilers, may register new
elements by calling the function register_element."""

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
# Modified by Marie E. Rognes <meg@simula.no>, 2010
# Modified by Lizao Li <lzlarryli@gmail.com>, 2015

from __future__ import print_function

from ufl.assertions import ufl_assert
from ufl.sobolevspace import L2, H1, H2, HDiv, HCurl, HEin
from ufl.common import istr

# List of valid elements
ufl_elements = {}

# Aliases: aliases[name] (...) -> (standard_name, ...)
aliases = {}

# Function for registering new elements
def register_element(family, short_name, value_rank, sobolev_space, mapping, degree_range, cellnames):
    "Register new finite element family"
    ufl_assert(family not in ufl_elements, 'Finite element \"%s\" has already been registered.' % family)
    ufl_elements[family]     = (family, short_name, value_rank, sobolev_space, mapping, degree_range, cellnames)
    ufl_elements[short_name] = (family, short_name, value_rank, sobolev_space, mapping, degree_range, cellnames)

def register_element2(family, value_rank, sobolev_space, mapping, degree_range, cellnames):
    "Register new finite element family"
    ufl_assert(family not in ufl_elements, 'Finite element \"%s\" has already been registered.' % family)
    ufl_elements[family] = (family, family, value_rank, sobolev_space, mapping, degree_range, cellnames)

def register_alias(alias, to):
    aliases[alias] = to

def show_elements():
    print("Showing all registered elements:")
    print("================================")
    shown = set()
    for k in sorted(ufl_elements.keys()):
        data = ufl_elements[k]
        if data in shown:
            continue
        shown.add(data)
        (family, short_name, value_rank, sobolev_space, mapping, degree_range, cellnames) = data
        print("Finite element family: %s, %s" % (repr(family), repr(short_name)))
        print("Sobolev space: %s" % (sobolev_space,))
        print("Mapping: %s" % (mapping,))
        print("Degree range: %s" % (degree_range,))
        print("Value rank: %s" % (value_rank,))
        print("Defined on cellnames: %s" % (cellnames,))
        print()

# FIXME: Consider cleanup of element names. Use notation from periodic table as the main, keep old names as compatibility aliases.

# NOTE: Any element with polynomial degree 0 will be considered L2, independent of the space passed to register_element.

# NOTE: The mapping of the element basis functions
#       from reference to physical representation is
#       chosen based on the sobolev space:
#       HDiv = contravariant Piola,
#       HCurl = covariant Piola,
#       H1/L2 = no mapping.

# TODO: If determining mapping from sobolev_space isn't sufficient
#       in the future, add mapping name as another element property.

# Cell groups
simplices = ("interval", "triangle", "tetrahedron")
cubes     = ("interval", "quadrilateral", "hexahedron")
any_cell  = (None,
             "vertex", "interval",
             "triangle", "tetrahedron",
             "quadrilateral", "hexahedron")

# Elements in the periodic table # TODO: Register these as aliases of periodic table element description instead of the other way around
register_element("Lagrange", "CG",                       0, H1,    "identity", (1, None), any_cell)                  # "P"
register_element("Brezzi-Douglas-Marini", "BDM",         1, HDiv,  "contravariant Piola", (1, None), simplices[1:])  # "BDMF" (2d), "N2F" (3d)
register_element("Discontinuous Lagrange", "DG",         0, L2,    "identity", (0, None), any_cell)                  # "DP"
register_element("Nedelec 1st kind H(curl)", "N1curl",   1, HCurl, "covariant Piola", (1, None), simplices[1:])      # "RTE"  (2d), "N1E" (3d)
register_element("Nedelec 2nd kind H(curl)", "N2curl",   1, HCurl, "covariant Piola", (1, None), simplices[1:])      # "BDME" (2d), "N2E" (3d)
register_element("Raviart-Thomas", "RT",                 1, HDiv,  "contravariant Piola", (1, None), simplices[1:])  # "RTF"  (2d), "N1F" (3d)

# Elements not in the periodic table
register_element("Argyris", "ARG",                       0, H2,   "identity", (1, None), simplices[1:])
register_element("Arnold-Winther", "AW",                 0, H1,   "identity", None, ("triangle",))
register_element("Brezzi-Douglas-Fortin-Marini", "BDFM", 1, HDiv, "contravariant Piola", (1, None), simplices[1:])
register_element("Crouzeix-Raviart", "CR",               0, L2,   "identity", (1, 1), simplices[1:])
# TODO: Implement generic Tear operator for elements instead of this:
register_element("Discontinuous Raviart-Thomas", "DRT",  1, L2,   "contravariant Piola", (1, None), simplices[1:])
register_element("Hermite", "HER",                       0, H1,   "identity", None, simplices[1:])
register_element("Mardal-Tai-Winther", "MTW",            0, H1,   "identity", None, ("triangle",))
register_element("Morley", "MOR",                        0, H2,   "identity", None, ("triangle",))

# Special elements
register_element("Boundary Quadrature", "BQ", 0, L2, "identity", (0, None), any_cell)
register_element("Bubble", "B",               0, H1, "identity", (2, None), simplices)
register_element("Quadrature", "Quadrature",  0, L2, "identity", (0, None), any_cell)
register_element("Real", "R",                 0, L2, "identity", (0, 0),    any_cell)
register_element("Undefined", "U",            0, L2, "identity", (0, None), any_cell)
register_element("Lobatto", "Lob",            0, L2, "identity", (1, None), ("interval",))
register_element("Radau",   "Rad",            0, L2, "identity", (0, None), ("interval",))
register_element("Regge", "Regge",            2, HEin, "pullback as metric", (0, None), simplices[1:])

# Let Nedelec H(div) elements be aliases to BDMs/RTs
register_alias("Nedelec 1st kind H(div)",
               lambda family, dim, order, degree: ("Raviart-Thomas", order))
register_alias("N1div",
               lambda family, dim, order, degree: ("Raviart-Thomas", order))

register_alias("Nedelec 2nd kind H(div)",
               lambda family, dim, order, degree: ("Brezzi-Douglas-Marini", order))
register_alias("N2div",
               lambda family, dim, order, degree: ("Brezzi-Douglas-Marini", order))

# New elements introduced for the periodic table 2014
register_element2("Q",     0, H1,    "identity",            (1, None), cubes)
register_element2("DQ",    0, L2,    "identity",            (0, None), cubes)
register_element2("RTCE",  1, HCurl, "covariant Piola",     (1, None), ("quadrilateral",))
register_element2("RTCF",  1, HDiv,  "contravariant Piola", (1, None), ("quadrilateral",))
register_element2("NCE",   1, HCurl, "covariant Piola",     (1, None), ("hexahedron",))
register_element2("NCF",   1, HDiv,  "contravariant Piola", (1, None), ("hexahedron",))

register_element2("S",     0, H1,    "identity",            (1, None), cubes)
register_element2("DPC",   0, L2,    "identity",            (1, None), cubes)
register_element2("BDMCE", 1, HCurl, "covariant Piola",     (1, None), ("quadrilateral",))
register_element2("BDMCF", 1, HDiv,  "contravariant Piola", (1, None), ("quadrilateral",))
register_element2("AAE",   1, HCurl, "covariant Piola",     (1, None), ("hexahedron",))
register_element2("AAF",   1, HDiv,  "contravariant Piola", (1, None), ("hexahedron",))

# New aliases introduced for the periodic table 2014
register_alias("P",    lambda family, dim, order, degree: ("Lagrange",                 order))
register_alias("DP",   lambda family, dim, order, degree: ("Discontinuous Lagrange",   order))
register_alias("RTE",  lambda family, dim, order, degree: ("Nedelec 1st kind H(curl)", order))
register_alias("RTF",  lambda family, dim, order, degree: ("Raviart-Thomas",           order))
register_alias("N1E",  lambda family, dim, order, degree: ("Nedelec 1st kind H(curl)", order))
register_alias("N1F",  lambda family, dim, order, degree: ("Raviart-Thomas",           order))

register_alias("BDME", lambda family, dim, order, degree: ("Nedelec 2nd kind H(curl)", order))
register_alias("BDMF", lambda family, dim, order, degree: ("Brezzi-Douglas-Marini",    order))
register_alias("N2E",  lambda family, dim, order, degree: ("Nedelec 2nd kind H(curl)", order))
register_alias("N2F",  lambda family, dim, order, degree: ("Brezzi-Douglas-Marini",    order))

# Finite element exterior calculus notation
def feec_element(family, n, r, k):
    # n = topological dimension of domain
    # r = polynomial order
    # k = form_degree

    # Note: We always map to edge elements in 2D, don't know how to differentiate otherwise?

    # Mapping from (feec name, domain dimension, form degree) to (family name, polynomial order)
    _feec_elements = {
        "P- Lambda": (
            (("P", r), ("DP", r - 1)),
            (("P", r), ("RTE", r),      ("DP", r - 1)),
            (("P", r), ("N1E", r),      ("N1F", r),   ("DP", r - 1)),
            ),
        "P Lambda": (
            (("P", r), ("DP", r)),
            (("P", r), ("BDME", r),     ("DP", r)),
            (("P", r), ("N2E", r),      ("N2F", r),   ("DP", r)),
            ),
        "Q- Lambda": (
            (("Q", r), ("DQ", r - 1)),
            (("Q", r), ("RTCE", r),     ("DQ", r - 1)),
            (("Q", r), ("NCE", r),      ("NCF", r),   ("DQ", r - 1)),
            ),
        "S Lambda": (
            (("S", r), ("DPC", r)),
            (("S", r), ("BDMCE", r),    ("DPC", r)),
            (("S", r), ("AAE", r),      ("AAF", r),   ("DPC", r)),
            ),
        }

    # New notation, old verbose notation (including "Lambda") might be removed
    _feec_elements["P-"] = _feec_elements["P- Lambda"]
    _feec_elements["P"]  = _feec_elements["P Lambda"]
    _feec_elements["Q-"] = _feec_elements["Q- Lambda"]
    _feec_elements["S"]  = _feec_elements["S Lambda"]

    family, r = _feec_elements[family][n - 1][k]

    return family, r

# General FEEC notation, old verbose (can be removed)
register_alias("P- Lambda", lambda family, dim, order, degree: feec_element(family, dim, order, degree))
register_alias("P Lambda",  lambda family, dim, order, degree: feec_element(family, dim, order, degree))
register_alias("Q- Lambda", lambda family, dim, order, degree: feec_element(family, dim, order, degree))
register_alias("S Lambda",  lambda family, dim, order, degree: feec_element(family, dim, order, degree))

# General FEEC notation, new compact notation
register_alias("P-", lambda family, dim, order, degree: feec_element(family, dim, order, degree))
#register_alias("P Lambda",  lambda family, dim, order, degree: feec_element(family, dim, order, degree))
register_alias("Q-", lambda family, dim, order, degree: feec_element(family, dim, order, degree))
#register_alias("S Lambda",  lambda family, dim, order, degree: feec_element(family, dim, order, degree))

def canonical_element_description(family, cell, order, form_degree):
    """Given basic element information, return corresponding element information on canonical form.

    Input: family, cell, (polynomial) order, form_degree
    Output: family (canonical), short_name (for printing), order, value shape, reference value shape, sobolev_space

    This is used by the FiniteElement constructor to ved input
    data against the element list and aliases defined in ufl.
    """

    # Get domain dimensions
    if cell is not None:
        tdim = cell.topological_dimension()
        gdim = cell.geometric_dimension()
        cellname = cell.cellname()
    else:
        tdim = None
        gdim = None
        cellname = None

    # Catch general FEEC notation "P" and "S"
    if form_degree is not None and family in ("P", "S"):
        family, order = feec_element(family, tdim, order, form_degree)

    # Check whether this family is an alias for something else
    while family in aliases:
        ufl_assert(tdim is not None, "Need dimension to handle element aliases.")
        (family, order) = aliases[family](family, tdim, order, form_degree)
        #info_blue("%s, is an alias for %s " % (
        #        (family, cell, order, form_degree),
        #        (name, dummy_cell, r)))

    # Check that the element family exists
    ufl_assert(family in ufl_elements,
               'Unknown finite element "%s".' % family)

    # Check that element data is valid (and also get common family name)
    (family, short_name, value_rank, sobolev_space, mapping, krange, cellnames) = ufl_elements[family]

    # Validate cellname if a valid cell is specified
    ufl_assert(cellname is None or cellname in cellnames,
        'Cellname "%s" invalid for "%s" finite element.' % (cellname, family))

    # Validate order if specified
    if order is not None:
        ufl_assert(krange is not None,
                   'Order "%s" invalid for "%s" finite element, '\
                       'should be None.' % (order, family))
        kmin, kmax = krange
        ufl_assert(kmin is None or order >= kmin,
                   'Order "%s" invalid for "%s" finite element.' %\
                       (order, family))
        ufl_assert(kmax is None or order <= kmax,
               'Order "%s" invalid for "%s" finite element.' %\
                       (istr(order), family))

    # Override sobolev_space for piecewise constants (TODO: necessary?)
    if order == 0:
        sobolev_space = L2
    if value_rank == 2:
        # Tensor valued fundamental elements in HEin have this shape
        ufl_assert(gdim is not None and tdim is not None,
               "Cannot infer shape of element without topological and geometric dimensions.")
        reference_value_shape = (tdim, tdim)
        value_shape = (gdim, gdim)
    elif value_rank == 1:
        # Vector valued fundamental elements in HDiv and HCurl have a shape
        ufl_assert(gdim != None and tdim != None,
               "Cannot infer shape of element without topological and geometric dimensions.")
        reference_value_shape = (tdim,)
        value_shape = (gdim,)
    elif value_rank == 0:
        # All other elements are scalar values
        reference_value_shape = ()
        value_shape = ()
    else:
        error("Invalid value rank %d." % value_rank)

    return family, short_name, order, value_shape, reference_value_shape, sobolev_space, mapping
