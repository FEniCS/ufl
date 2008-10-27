"""This module defines utilities working with variables
in UFL expressions, either by inserting variables in an
expression or extracting information about variables in
an expression."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-05-07 -- 2008-10-27"

from ..output import ufl_assert, ufl_error, ufl_warning

# Classes:
from ..base import UFLObject, FloatValue, ZeroType 
from ..indexing import MultiIndex
from ..variable import Variable

# Other algorithms:
from .traversal import post_traversal
from .analysis import extract_basisfunctions, extract_coefficients, extract_indices
from .transformations import ufl_reuse_handlers, transform, transform_integrands


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

def extract_duplications(expression):
    "Build a set of all repeated expressions in expression."
    ufl_assert(isinstance(expression, UFLObject), "Expecting UFL expression.")
    handled = set()
    duplicated = set()
    for (o, stack) in post_traversal(expression):
        if o in handled:
            duplicated.add(o)
        handled.add(o)
    return duplicated

def _mark_duplications(expression, handlers, variables, dups):
    """Wrap subexpressions that are equal (completely equal, not mathematically equivalent)
    in Variable objects to facilitate subexpression reuse."""
    
    # TODO: Indices will often mess this up.
    # Can we renumber indices consistently from the leaves to avoid that problem?
    # This may introduce many ComponentTensor/Indexed objects for relabeling of indices though.
    # Probably need some kind of pattern matching to make this effective.
    # 
    # What this does do well is insert Variables around subexpressions that the
    # user actually identified manually in his code like in "a = ...; b = a*(1+a)",
    # and expressions without indices (prior to expand_compounds).
    
    # check variable cache
    var = variables.get(expression, None)
    if var is not None:
        return var
    
    # skip some types
    _skiptypes = (MultiIndex, FloatValue, ZeroType)
    if isinstance(expression, _skiptypes):
        return expression
    
    # handle subexpressions
    ops = [_mark_duplications(o, handlers, variables, dups) for o in expression.operands()]
    
    # get handler
    c = expression._uflid
    if c in handlers:
        h = handlers[c]
    else:
        ufl_error("Didn't find class %s among handlers." % c)
    
    # transform subexpressions
    handled = h(expression, *ops)
    
    # wrap in variable if a duplicate
    if expression in dups or handled in dups: # TODO: Not sure if it is necessary to look for handled
        if not isinstance(handled, Variable):
            handled = Variable(handled)
        variables[expression] = handled
        variables[handled] = handled
    
    return handled

def mark_duplications(expression):
    "Wrap all duplicated expressions as Variables."
    dups = extract_duplications(expression) # FIXME: Maybe avoid iteration into variables in extract_duplications and handle variables explicitly in here?
    variables = {}
    vars = extract_variables(expression)
    for v in vars:
        variables[v._expression] = v
    d = ufl_reuse_handlers()
    return _mark_duplications(expression, d, variables, dups)
