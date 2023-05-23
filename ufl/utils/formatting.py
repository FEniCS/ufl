# -*- coding: utf-8 -*-
"""Various string formatting utilities."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs and Anders Logg
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


def camel2underscore(name):
    """Convert a CamelCaps string to underscore_syntax."""
    letters = []
    lastlower = False
    for l in name:
        thislower = l.islower()
        if not thislower:
            # Don't insert _ between multiple upper case letters
            if lastlower:
                letters.append("_")
            l = l.lower()  # noqa: E741
        lastlower = thislower
        letters.append(l)
    return "".join(letters)


def lstr(l):
    """Pretty-print list or tuple, invoking str() on items instead of repr() like str() does."""
    if isinstance(l, list):
        return "[" + ", ".join(lstr(item) for item in l) + "]"
    elif isinstance(l, tuple):
        return "(" + ", ".join(lstr(item) for item in l) + ")"
    return str(l)


def tstr(t, colsize=80):
    """Pretty-print list of tuples of key-value pairs."""
    if not t:
        return ""

    # Compute maximum key length
    keylen = max(len(str(item[0])) for item in t)

    # Key-length cannot be larger than colsize
    if keylen > colsize:
        return str(t)

    # Pretty-print table
    s = ""
    for (key, value) in t:
        key = str(key)
        if isinstance(value, str):
            value = "'%s'" % value
        else:
            value = str(value)
        s += key + ":" + " " * (keylen - len(key) + 1)
        space = ""
        while len(value) > 0:
            end = min(len(value), colsize - keylen)
            s += space + value[:end] + "\n"
            value = value[end:]
            space = " " * (keylen + 2)
    return s


def sstr(s):
    """Pretty-print set."""
    return ", ".join(str(x) for x in s)


def istr(o):
    """Format object as string, inserting ? for None."""
    if o is None:
        return "?"
    else:
        return str(o)


def estr(elements):
    """Format list of elements for printing."""
    return ", ".join(e.shortstr() for e in elements)


def _indent_string(n):
    return "    " * n


def _tree_format_expression(expression, indentation, parentheses):
    ind = _indent_string(indentation)
    if expression._ufl_is_terminal_:
        s = "%s%s" % (ind, repr(expression))
    else:
        sops = [_tree_format_expression(o, indentation + 1, parentheses) for o in expression.ufl_operands]
        s = "%s%s\n" % (ind, expression._ufl_class_.__name__)
        if parentheses and len(sops) > 1:
            s += "%s(\n" % (ind,)
        s += "\n".join(sops)
        if parentheses and len(sops) > 1:
            s += "\n%s)" % (ind,)
    return s


def tree_format(expression, indentation=0, parentheses=True):
    from ufl.core.expr import Expr
    from ufl.form import Form
    from ufl.integral import Integral

    s = ""

    if isinstance(expression, Form):
        form = expression
        integrals = form.integrals()
        integral_types = sorted(set(itg.integral_type() for itg in integrals))
        itgs = []
        for integral_type in integral_types:
            itgs += list(form.integrals_by_type(integral_type))

        ind = _indent_string(indentation)
        s += ind + "Form:\n"
        s += "\n".join(tree_format(itg, indentation + 1, parentheses) for itg in itgs)

    elif isinstance(expression, Integral):
        ind = _indent_string(indentation)
        s += ind + "Integral:\n"
        ind = _indent_string(indentation + 1)
        s += ind + "integral type: %s\n" % expression.integral_type()
        s += ind + "subdomain id: %s\n" % expression.subdomain_id()
        s += ind + "integrand:\n"
        s += tree_format(expression._integrand, indentation + 2, parentheses)

    elif isinstance(expression, Expr):
        s += _tree_format_expression(expression, indentation, parentheses)

    else:
        raise ValueError(f"Invalid object type {type(expression)}")

    return s
