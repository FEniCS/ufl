# -*- coding: utf-8 -*-
"""Signature computation for forms."""

# Copyright (C) 2012-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import hashlib
from ufl.classes import (Label,
                         Index, MultiIndex,
                         Coefficient, Argument,
                         GeometricQuantity, ConstantValue, Constant,
                         ExprList, ExprMapping)
from ufl.log import error
from ufl.corealg.traversal import traverse_unique_terminals, pre_traversal
from ufl.algorithms.domain_analysis import canonicalize_metadata


def compute_multiindex_hashdata(expr, index_numbering):
    data = []
    for i in expr:
        if isinstance(i, Index):
            j = index_numbering.get(i)
            if j is None:
                # Use negative ints for Index
                j = -(len(index_numbering) + 1)
                index_numbering[i] = j
            data.append(j)
        else:
            # Use nonnegative ints for FixedIndex
            data.append(int(i))
    return tuple(data)


def compute_terminal_hashdata(expressions, renumbering):

    if not isinstance(expressions, list):
        expressions = [expressions]
    assert renumbering is not None

    # Extract a unique numbering of free indices, as well as form
    # arguments, and just take repr of the rest of the terminals while
    # we're iterating over them
    terminal_hashdata = {}
    labels = {}
    index_numbering = {}
    for expression in expressions:
        for expr in traverse_unique_terminals(expression):

            if isinstance(expr, MultiIndex):
                # Indices need a canonical numbering for a stable
                # signature, thus this algorithm
                data = compute_multiindex_hashdata(expr, index_numbering)

            elif isinstance(expr, ConstantValue):
                data = expr._ufl_signature_data_(renumbering)

            elif isinstance(expr, Coefficient):
                data = expr._ufl_signature_data_(renumbering)

            elif isinstance(expr, Constant):
                data = expr._ufl_signature_data_(renumbering)

            elif isinstance(expr, Argument):
                data = expr._ufl_signature_data_(renumbering)

            elif isinstance(expr, GeometricQuantity):
                data = expr._ufl_signature_data_(renumbering)

            elif isinstance(expr, Label):
                # Numbering labels as we visit them # TODO: Include in
                # renumbering
                data = labels.get(expr)
                if data is None:
                    data = "L%d" % len(labels)
                    labels[expr] = data

            elif isinstance(expr, ExprList):
                # Not really a terminal but can have 0 operands...
                data = "[]"

            elif isinstance(expr, ExprMapping):
                # Not really a terminal but can have 0 operands...
                data = "{}"

            else:
                error("Unknown terminal type %s" % type(expr))

            terminal_hashdata[expr] = data

    return terminal_hashdata


def compute_expression_hashdata(expression, terminal_hashdata):
    # The hashdata computed here can be interpreted as prefix operator
    # notation, i.e. we store the equivalent of '+ * a b * c d' for
    # the expression (a*b)+(c*d)
    expression_hashdata = []
    for expr in pre_traversal(expression):
        if expr._ufl_is_terminal_:
            data = terminal_hashdata[expr]
        else:
            data = expr._ufl_typecode_  # TODO: Use expr._ufl_signature_data_()? More extensible, but more overhead.
        expression_hashdata.append(data)
    # Oneliner: TODO: Benchmark, maybe use a generator?
    # expression_hashdata = [(terminal_hashdata[expr] if expr._ufl_is_terminal_ else expr._ufl_typecode_)
    #                       for expr in pre_traversal(expression)]
    return expression_hashdata


def compute_expression_signature(expr, renumbering):  # FIXME: Fix callers
    # FIXME: Rewrite in terms of compute_form_signature?

    # Build hashdata for all terminals first
    terminal_hashdata = compute_terminal_hashdata([expr], renumbering)

    # Build hashdata for full expression
    expression_hashdata = compute_expression_hashdata(expr, terminal_hashdata)

    # Pass it through a seriously overkill hashing algorithm
    # (should we use sha1 instead?)
    data = str(expression_hashdata).encode("utf-8")
    return hashlib.sha512(data).hexdigest()


def compute_form_signature(form, renumbering):  # FIXME: Fix callers
    # Extract integrands
    integrals = form.integrals()
    integrands = [integral.integrand() for integral in integrals]

    # Build hashdata for all terminals first, with on-the-fly
    # replacement of functions and index labels.
    terminal_hashdata = compute_terminal_hashdata(integrands, renumbering)

    # Build hashdata for each integral
    hashdata = []
    for integral in integrals:
        # Compute hash data for expression, this is the expensive part
        integrand_hashdata = compute_expression_hashdata(integral.integrand(),
                                                         terminal_hashdata)

        domain_hashdata = integral.ufl_domain()._ufl_signature_data_(renumbering)

        # Collect all data about integral that should be reflected in
        # signature, including compiler data but not domain data,
        # because compiler data affects the way the integral is
        # compiled while domain data is only carried for convenience
        # in the problem solving environment.
        integral_hashdata = (
            integrand_hashdata,
            domain_hashdata,
            integral.integral_type(),
            integral.subdomain_id(),
            canonicalize_metadata(integral.metadata()),
        )

        hashdata.append(integral_hashdata)

    # Pass it through a seriously overkill hashing algorithm
    # (should we use sha1 instead?)
    data = str(hashdata).encode("utf-8")
    return hashlib.sha512(data).hexdigest()
