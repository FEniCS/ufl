
from __future__ import absolute_import

# TODO: Clean up organization of algorithms.
from .traversal import iter_expressions, post_traversal, pre_traversal, post_walk, pre_walk, walk
from .analysis import extract_type, basisfunctions, coefficients, elements, unique_elements, classes, variables, duplications, value_shape, domain
from .predicates import is_multilinear
from .checks import validate_form
from .utilities import load_forms, integral_info, form_info

from .transformations import ufl2ufl, ufl2latex, flatten, expand_compounds, transform_integrands, strip_variables, strip_variables2

from .transformations_work_in_progress import renumber_indices, renumber_basisfunctions, renumber_functions, substitute_indices, expand_indices, discover_indices

