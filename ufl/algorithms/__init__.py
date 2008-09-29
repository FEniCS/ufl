"This module collects algorithms and utility functions operating on UFL objects."

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-14 -- 2008-09-24"

# Utilities for traversing over expression trees in different ways
from .traversal import iter_expressions, post_traversal, pre_traversal, post_walk, pre_walk, walk

# Utilities for extracting information from forms and expressions
from .analysis import extract_type, classes, domain, value_shape
from .analysis import basisfunctions, coefficients, elements, unique_elements
from .analysis import variables, duplications
from .analysis import monomials
from .formdata import FormData

# Utilities for checking properties of forms
from .predicates import is_multilinear

# Utilities for error checking of forms
from .checks import validate_form

# Utilites for modifying expressions and forms
from .transformations import ufl2ufl, ufl2uflcopy, transform_integrands
from .transformations import renumber_indices
from .transformations import expand_compounds, flatten, strip_variables
from .transformations import substitute_indices, expand_indices
from .transformations import split_by_dependencies, mark_duplications
from .transformations import replace, replace_in_form
from .transformations import compute_form_transpose, compute_form_action
from .transformations import compute_form_lhs, compute_form_rhs
from .transformations import compute_dirichlet_functional, compute_dual_form

# Utilities for Automatic Functional Differentiation
from .ad import compute_form_derivative

# Utilities for UFL object printing
from .ufl2latex import ufl2latex
from .printing import integral_info, form_info, tree_format

# Utilities for form file handling
from .formfiles import load_forms
