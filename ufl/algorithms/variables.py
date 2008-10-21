"""This module defines utilities working with variables
in UFL expressions, either by inserting variables in an
expression or extracting information about variables in
an expression."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-05-07 -- 2008-10-21"

from ..output import ufl_assert, ufl_error, ufl_warning

# Classes:
from ..variable import Variable

# Other algorithms:
from .analysis import extract_basisfunctions, extract_coefficients, extract_indices, extract_duplications
from .transformations import ufl_reuse_handlers, transform, transform_integrands


def _mark_duplications(expression, handlers, variables, dups):
    """Convert a UFLExpression according to rules defined by
    the mapping handlers = dict: class -> conversion function."""
    
    # check variable cache
    var = variables.get(expression, None)
    if var is not None:
        return var
    
    # handle subexpressions
    ops = [_mark_duplications(o, handlers, variables, dups) for o in expression.operands()]
    
    # get handler
    c = type(expression)
    if c in handlers:
        h = handlers[c]
    else:
        ufl_error("Didn't find class %s among handlers." % c)
    
    # transform subexpressions
    handled = h(expression, *ops)
    
    if expression in dups:
        handled = Variable(handled)
        variables[expression] = handled
    
    return handled


def mark_duplications(expression):
    "Wrap all duplicated expressions as Variables."
    dups = duplications(expression)
    variables = {}
    d = ufl_reuse_handlers()
    return _mark_duplications(expression, d, variables, dups)


def strip_variables(expression, handled_variables=None):
    d = ufl_reuse_handlers()
    if handled_variables is None:
        handled_variables = {}
    def s_variable(x):
        if x._count in handled_variables:
            return handled_variables[x._count]
        v = strip_variables(x._expression, handled_variables)
        handled_variables[x._count] = v
        return v
    d[Variable] = s_variable
    return transform(expression, d)


def extract_variables(expression, handled_expressions=None):
    if handled_expressions is None:
        handled_expressions = set()
    vars = []
    i = id(expression)
    if i in handled_expressions:
        return vars
    handled_expressions.add(i)
    if isinstance(expression, Variable):
        vars.extend(extract_variables(expression._expression, handled_expressions))
        vars.append(expression)
    else:
        for o in expression.operands():
            vars.extend(extract_variables(o, handled_expressions))
    return vars 
