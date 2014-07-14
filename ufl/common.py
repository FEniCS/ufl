"This module contains a collection of common utilities."

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
# Modified by Kristian Oelgaard, 2009

# TODO: These things used to reside here, if we import from ufl.utils instead where applicable we can remove common.py

from ufl.utils.indexflattening import shape_to_strides, unflatten_index, flatten_multiindex
from ufl.utils.sequences import product, unzip, xor, or_tuples, and_tuples, iter_tree, recursive_chain
from ufl.utils.traversal import (fast_pre_traversal, fast_pre_traversal2,
                                 unique_pre_traversal, unique_post_traversal,
                                 fast_post_traversal, fast_post_traversal2)
from ufl.utils.formatting import lstr, estr, istr, sstr, tstr, dstr, camel2underscore
from ufl.utils.dicts import split_dict, slice_dict, mergedicts, mergedicts2, subdict, dict_sum, EmptyDictType, EmptyDict
from ufl.utils.counted import counted_init, ExampleCounted
from ufl.utils.timer import Timer
from ufl.utils.stacks import Stack, StackDict
from ufl.utils.ufltypedicts import UFLTypeDict, UFLTypeDefaultDict
from ufl.utils.sorting import topological_sorting, sorted_by_count, sorted_by_key
from ufl.utils.system import get_status_output, openpdf, pdflatex, write_file
