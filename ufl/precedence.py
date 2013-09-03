"Precedence handling."

# Copyright (C) 2008-2013 Martin Sandve Alnes
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
# First added:  2009-03-27
# Last changed: 2011-06-07

from ufl.log import warning

def parstr(child, parent, pre="(", post=")", format=str):
    # Execute when needed instead of on import,
    # which leads to all kinds of circular trouble.
    # Fixing this could be an optimization of str(expr) though.
    if not hasattr(parent, '_precedence'):
        assign_precedences(build_precedence_list())

    # We want child to be evaluated fully first,
    # and if the parent has higher precedence
    # we later wrap in ().
    s = format(child)

    # Operators where operands are always parenthesized
    if parent._precedence == 0:
        return pre + s + post

    # If parent operator binds stronger than child, must parenthesize child
    # FIXME: Is this correct for all possible positions of () in a + b + c?
    if parent._precedence > child._precedence:
        return pre + s + post

    # Nothing needed
    return s

def build_precedence_list():
    from ufl.classes import Operator, Terminal, Sum, IndexSum, Product, Division, Power, MathFunction, Abs

    # TODO: Fill in other types...
    #Power <= Indexed
    #Power <= Transposed

    precedence_list = []
    # Default behaviour: should always add parentheses
    precedence_list.append((Operator,))

    precedence_list.append((Sum,))

    # sum_i a + b = (sum_i a) + b != sum_i (a + b), sum_i binds stronger than +, but weaker than product
    precedence_list.append((IndexSum,))

    precedence_list.append((Product, Division,))

    # NB! Depends on language!
    precedence_list.append((Power, MathFunction, Abs))

    precedence_list.append((Terminal,))
    return precedence_list

def build_precedence_mapping(precedence_list):
    """Given a precedence list, build a dict with class->int mappings.
    Utility function used by some external code.
    """
    from ufl.classes import Expr, all_ufl_classes, abstract_classes
    pm = {}
    missing = set()
    # Assign integer values for each precedence level
    k = 0
    for p in precedence_list:
        for c in p:
            pm[c] = k
        k += 1
    # Check for missing classes, fill in subclasses
    for c in all_ufl_classes:
        if c not in abstract_classes and c not in pm:
            b = c.__bases__[0]
            while b is not Expr:
                if b in pm:
                    pm[c] = pm[b]
                    break
                b = b.__bases__[0]
            if c not in pm:
                missing.add(c)
    return pm, missing

def assign_precedences(precedence_list):
    "Given a precedence list, assign ints to class._precedence."
    pm, missing = build_precedence_mapping(precedence_list)
    for c, p in sorted(pm.iteritems(), key=lambda x: x[0].__name__):
        c._precedence = p
    if missing:
        msg = "Missing precedence levels for classes:\n" +\
            "\n".join('  %s' % c for c in sorted(missing))
        warning(msg)
