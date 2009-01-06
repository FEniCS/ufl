"This module collects algorithms and utility functions operating on UFL objects."

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-14 -- 2009-01-05"

# Modified by Anders Logg, 2008

# Utilities for traversing over expression trees in different ways
from ufl.algorithms.traversal import iter_expressions, traverse_terminals, \
                                     post_traversal, pre_traversal, \
                                     post_walk, pre_walk, walk

# Utilities for extracting information from forms and expressions
from ufl.algorithms.analysis import extract_classes, extract_type, \
                                    extract_basisfunctions, extract_coefficients, \
                                    extract_elements, extract_unique_elements, \
                                    extract_variables, extract_duplications
from ufl.algorithms.monomials import extract_monomials

# Utility class for easy collecting of data about form
from ufl.algorithms.formdata import FormData

# Utilities for checking properties of forms
from ufl.algorithms.predicates import is_multilinear

# Utilities for error checking of forms
from ufl.algorithms.checks import validate_form

# Utilites for modifying expressions and forms
from ufl.algorithms.transformations import transform, \
                                           ufl2ufl, ufl2uflcopy, \
                                           replace, flatten, strip_variables, \
                                           expand_compounds, mark_duplications

# Utilities for working with dependencies of subexpressions
from ufl.algorithms.dependencies import split_by_dependencies

# Utilities for transforming complete Forms into other Forms
from ufl.algorithms.formtransformations import compute_form_adjoint, compute_form_action, \
                                               compute_form_lhs, compute_form_rhs #, compute_dirichlet_functional

# Utilities for Automatic Functional Differentiation
from ufl.algorithms.ad import expand_derivatives, compute_diff, propagate_spatial_derivatives, compute_form_derivative

# Utilities for UFL object printing
from ufl.algorithms.printing import integral_info, form_info, tree_format
from ufl.algorithms.ufl2latex import ufl2latex, ufl2tex, ufl2pdf, forms2latexdocument
from ufl.algorithms.ufl2dot import ufl2dot

# Utilities for form file handling
from ufl.algorithms.formfiles import load_forms

# State of files (in the opinion of Martin):
#    traversal.py           - Ok.
#    analysis.py            - Ok, some unused stuff.
#    formdata.py            - Probably don't need both self.unique_elements and self.sub_elements?
#                             Need to improve quadrature order estimation.
#    predicates.py          - is_multilinear seems ok but needs testing, extract_monomials is just a sketch.
#    checks.py              - Ok, more checks are welcome.
#    formfiles.py           - Ok.
#    transformations.py     - Ok.
#    monomials.py           - Not working. FFC specific?
#    dependencies.py        - Rework with computational graphs!
#    formtransformations.py - Ok? Needs testing.
#    ad.py                  - In horrible state. Rework!
#    printing.py            - Ok.
#    latextools.py          - Ok.
#    ufl2latex.py           - Fix precedence stuff.
#    ufl2dot.py             - Rework with graph tools.

