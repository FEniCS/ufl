"""Signature computation for forms."""

# Copyright (C) 2012-2013 Martin Sandve Alnes
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
# First added:  2012-03-29
# Last changed: 2012-04-12

import hashlib
from ufl.classes import Index, MultiIndex, Coefficient, Argument, Terminal, Label
from ufl.log import error
from ufl.algorithms.traversal import traverse_terminals2
from ufl.common import fast_pre_traversal, sorted_by_count

def compute_multiindex_hashdata(expr, index_numbering):
    data = []
    for i in expr:
        if isinstance(i, Index):
            j = index_numbering.get(i)
            if j is None:
                # Use negative ints for Index
                j = -(len(index_numbering)+1)
                index_numbering[i] = j
            data.append(j)
        else:
            # Use nonnegative ints for FixedIndex
            data.append(int(i))
    return data

def compute_terminal_hashdata(expressions, function_replace_map=None):
    if not isinstance(expressions, list):
        expressions = [expressions]
    if function_replace_map is None:
        function_replace_map = {}

    # Extract a unique numbering of free indices,
    # as well as form arguments, and just take
    # repr of the rest of the terminals while
    # we're iterating over them
    terminal_hashdata = {}
    labels = {}
    index_numbering = {}
    coefficients = set()
    arguments = set()
    for expression in expressions:
        for expr in traverse_terminals2(expression):
            if isinstance(expr, MultiIndex):
                terminal_hashdata[expr] = compute_multiindex_hashdata(expr,
                                                                      index_numbering)
            elif isinstance(expr, Coefficient):
                coefficients.add(expr)
            elif isinstance(expr, Argument):
                arguments.add(expr)
            elif isinstance(expr, Label):
                data = labels.get(expr)
                if data is None:
                    data = "L%d" % len(labels)
                    labels[expr] = data
                terminal_hashdata[expr] = data
            else:
                terminal_hashdata[expr] = repr(expr)

    # Apply renumbering of form arguments
    # (Note: some duplicated work here and in preprocess,
    # to allow using this function without preprocess.)
    coefficients = sorted_by_count(coefficients)
    arguments = sorted_by_count(arguments)
    for i, e in enumerate(coefficients):
        er = function_replace_map.get(e) or e.reconstruct(count=i)
        terminal_hashdata[e] = repr(er)
    for i, e in enumerate(arguments):
        er = function_replace_map.get(e) or e.reconstruct(count=i)
        terminal_hashdata[e] = repr(er)

    return terminal_hashdata

def compute_expression_hashdata(expression, terminal_hashdata):
    # The hashdata computed here can be interpreted as
    # prefix operator notation, i.e. we store the equivalent
    # of '+ * a b * c d' for the expression (a*b)+(c*d)
    expression_hashdata = []
    for expr in fast_pre_traversal(expression):
        if isinstance(expr, Terminal):
            data = terminal_hashdata[expr]
        else:
            data = expr._classid
        expression_hashdata.append(data)
    return expression_hashdata

def compute_expression_signature(expr, function_replace_map=None):
    # Build hashdata for all terminals first
    terminal_hashdata = compute_terminal_hashdata([expr],
                                                  function_replace_map)

    # Build hashdata for full expression
    expression_hashdata = compute_expression_hashdata(expr,
                                                      terminal_hashdata)

    # Pass it through a seriously overkill hashing algorithm :) TODO: How fast is this? Reduce?
    return hashlib.sha512(str(expression_hashdata)).hexdigest()

def compute_form_signature(form, function_replace_map=None):
    integrals = form.integrals()
    integrands = [integral.integrand() for integral in integrals]

    # Build hashdata for all terminals first, with on-the-fly
    # replacement of functions and index labels.
    terminal_hashdata = compute_terminal_hashdata(integrands,
                                                  function_replace_map)
    # Build hashdata for each integral
    hashdata = []
    for integral in integrals:
        # Compute hash data for expression, this is the expensive part
        expression_hashdata = compute_expression_hashdata(integral.integrand(),
                                                          terminal_hashdata)

        # Collect all data about integral that should be reflected in signature,
        # including compiler data but not domain data, because compiler data
        # affects the way the integral is compiled while domain data is only
        # carried for convenience in the problem solving environment.
        integral_hashdata = (expression_hashdata,
                             integral.domain_type(),
                             integral.domain_id(),
                             repr(integral.compiler_data()),
                             )

        hashdata.append(integral_hashdata)

    # Pass hashdata through a seriously overkill hashing algorithm :) TODO: How fast is this? Reduce?
    return hashlib.sha512(str(hashdata)).hexdigest()
