"This module collects algorithms and utility functions operating on UFL objects."

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-14 -- 2008-08-21"

# Utilities for traversing over expression trees in different ways
from .traversal import iter_expressions, post_traversal, pre_traversal, post_walk, pre_walk, walk

# Utilities for extracting information from forms and expressions
from .analysis import extract_type, classes, domain, value_shape
from .analysis import basisfunctions, coefficients, elements, unique_elements
from .analysis import variables, duplications

# Utilities for checking properties of forms
from .predicates import is_multilinear

# Utilities for error checking of forms
from .checks import validate_form

# Utilities for form file handling
from .utilities import load_forms, integral_info, form_info

# Utilites for modifying expressions and forms
from .transformations import ufl2ufl, ufl2latex, transform_integrands
from .transformations import renumber_indices, renumber_arguments
from .transformations import expand_compounds, flatten, strip_variables
from .transformations import substitute_indices, expand_indices
from .transformations import split_by_dependencies
