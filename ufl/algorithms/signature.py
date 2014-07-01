"""Signature computation for forms."""

# Copyright (C) 2012-2014 Martin Sandve Alnes
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

import hashlib
from ufl.classes import Index, MultiIndex, Coefficient, Argument, Terminal, Label, FormArgument, GeometricQuantity, ConstantValue
from ufl.log import error
from ufl.algorithms.traversal import traverse_unique_terminals
from ufl.common import fast_pre_traversal, sorted_by_count
from ufl.geometry import join_domains

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
    return tuple(data)

def compute_terminal_hashdata(expressions, domain_numbering, function_replace_map=None):
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
    for expression in expressions:
        for expr in traverse_unique_terminals(expression):

            if isinstance(expr, MultiIndex):
                # Indices need a canonical numbering for a stable signature, thus this algorithm
                data = compute_multiindex_hashdata(expr, index_numbering)

            elif isinstance(expr, ConstantValue):
                # For literals no renumbering is necessary
                # TODO: This may change if we annotate literals with an Argument
                data = expr.signature_data()

            elif isinstance(expr, Coefficient):
                # Save coefficients for renumbering in next phase
                coefficients.add(expr)
                continue

            elif isinstance(expr, Argument):
                # With the new design where we specify argument number explicitly we avoid renumbering
                data = expr.signature_data(domain_numbering=domain_numbering)

            elif isinstance(expr, GeometricQuantity):
                # Assuming all geometric quantities are defined by just their class + domain
                data = expr.signature_data(domain_numbering=domain_numbering)

            elif isinstance(expr, Label):
                # Numbering labels as we visit them
                data = labels.get(expr)
                if data is None:
                    data = "L%d" % len(labels)
                    labels[expr] = data

            else:
                error("Unknown terminal type %s" % type(expr))

            terminal_hashdata[expr] = data

    # Apply renumbering of coefficients
    # (Note: some duplicated work here and in preprocess, to
    # allow using this function without full preprocessing.)
    for i, e in enumerate(sorted_by_count(coefficients)):
        er = function_replace_map.get(e)
        if er is None:
            er = e
        data = er.signature_data(count=i, domain_numbering=domain_numbering)
        terminal_hashdata[e] = data

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

def build_domain_numbering(domains):
    # Create canonical numbering of domains for stable signature
    # (ordering defined by __lt__ implementation in Domain class)
    assert None not in domains

    # Collect domain keys
    items = []
    for i, domain in enumerate(domains):
        key = (domain.cell(), domain.label())
        items.append((key, i))

    # Build domain numbering, not allowing repeated keys
    domain_numbering = {}
    for key, i in items:
        if key in domain_numbering:
            error("Domain key %s occured twice!" % (key,))
        domain_numbering[key] = i

    # Build domain numbering extension for None-labeled domains, not allowing ambiguity
    from collections import defaultdict
    domain_numbering2 = defaultdict(list)
    for key, i in items:
        cell, label = key
        key2 = (cell, None)
        domain_numbering2[key2].append(domain_numbering[key])

    # Add None-based key only where unambiguous
    for key, i in items:
        cell, label = key
        key2 = (cell, None)
        if len(domain_numbering2[key2]) == 1:
            domain_numbering[key2] = domain_numbering[key]
        else:
            # Two domains occur with same properties but different label,
            # so we cannot decide which one to map None-labeled Domains to.
            pass

    return domain_numbering

def compute_expression_signature(expr, function_replace_map=None):
    # Build a stable numbering of absolutely all domains from expr
    domain_numbering = build_domain_numbering(list(expr.domains()))

    # Build hashdata for all terminals first
    terminal_hashdata = compute_terminal_hashdata([expr], domain_numbering,
                                                  function_replace_map)

    # Build hashdata for full expression
    expression_hashdata = compute_expression_hashdata(expr,
                                                      terminal_hashdata)

    # Pass it through a seriously overkill hashing algorithm :) TODO: How fast is this? Reduce?
    return hashlib.sha512(str(expression_hashdata).encode('utf-8')).hexdigest()

def compute_form_signature(form, function_replace_map=None):
    # Extract integrands
    integrals = form.integrals()
    integrands = [integral.integrand() for integral in integrals]

    # Extract absolutely all domains from form
    all_domains = list(form.domains())
    for integrand in integrands:
        all_domains.extend(integrand.domains())
    domains = join_domains(all_domains)

    # Build a stable numbering of absolutely all domains from form
    domain_numbering = build_domain_numbering(domains)

    # Build hashdata for all terminals first, with on-the-fly
    # replacement of functions and index labels.
    terminal_hashdata = compute_terminal_hashdata(integrands, domain_numbering,
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
                             integral.domain().signature_data(domain_numbering=domain_numbering),
                             integral.integral_type(),
                             integral.subdomain_id(),
                             repr(integral.metadata()),
                             )

        hashdata.append(integral_hashdata)

    # Pass hashdata through a seriously overkill hashing algorithm :) TODO: How fast is this? Reduce?
    return hashlib.sha512(str(hashdata).encode('utf-8')).hexdigest()
