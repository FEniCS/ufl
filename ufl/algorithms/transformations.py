"""This module defines expression transformation utilities,
either converting UFL expressions to new UFL expressions or
converting UFL expressions to other representations."""

# Copyright (C) 2008-2013 Martin Sandve Alnes and Anders Logg
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
# Modified by Anders Logg, 2009-2010
#
# First added:  2008-05-07
# Last changed: 2012-04-12

# --- BEGIN dummy imports to keep imports in external code working:
from ufl.algorithms.multifunction import MultiFunction
from ufl.algorithms.transformer import Transformer, is_post_handler
from ufl.algorithms.transformer import transform, transform_integrands, apply_transformer
from ufl.algorithms.transformer import ReuseTransformer, ufl2ufl
from ufl.algorithms.transformer import CopyTransformer, ufl2uflcopy
from ufl.algorithms.transformer import VariableStripper, strip_variables
from ufl.algorithms.replace import Replacer, replace
from ufl.algorithms.expand_compounds import CompoundExpander, expand_compounds
from ufl.algorithms.estimate_degrees import SumDegreeEstimator, estimate_total_polynomial_degree
from ufl.algorithms.argument_dependencies import ArgumentDependencyExtracter, extract_argument_dependencies, NotMultiLinearException
from ufl.algorithms.deprecated import TreeFlattener, flatten
from ufl.algorithms.deprecated import DuplicationMarker, mark_duplications
from ufl.algorithms.deprecated import DuplicationPurger, purge_duplications
# --- END dummy imports to keep imports in external code working.
