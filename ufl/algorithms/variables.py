"""This module defines utilities working with variables
in UFL expressions, either by inserting variables in an
expression or extracting information about variables in
an expression."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-05-07 -- 2008-10-29"

from ..output import ufl_assert, ufl_error, ufl_warning
from ..common import UFLTypeDict

# Classes:
from ..base import Expr
from ..zero import Zero
from ..scalar import FloatValue, IntValue 
from ..indexing import MultiIndex
from ..variable import Variable
from ..classes import FacetNormal, Identity
from ..classes import ufl_classes

# Other algorithms:
from .traversal import post_traversal
from .analysis import extract_basisfunctions, extract_coefficients, extract_indices
from .transformations import ufl_reuse_handlers, transform, transform_integrands

def strip_variables(expression, handled_variables=None):
    if handled_variables is None:
        handled_variables = {}
    d = ufl_reuse_handlers()
    def s_variable(x):
        v = handled_variables.get(x._count, None)
        if v is not None: return v
        v = strip_variables(x._expression, handled_variables)
        handled_variables[x._count] = v
        return v
    d[Variable] = s_variable
    return transform(expression, d)

def extract_variables(expression, handled_vars=None):
    if handled_vars is None:
        handled_vars = set()
    if isinstance(expression, Variable):
        i = expression._count
        if i in handled_vars:
            return []
        handled_vars.add(i)
        vars = []
        vars.extend(extract_variables(expression._expression, handled_vars))
        vars.append(expression)
    else:
        vars = []
        for o in expression.operands():
            vars.extend(extract_variables(o, handled_vars))
    return vars 

def extract_duplications(expression):
    "Build a set of all repeated expressions in expression."
    ufl_assert(isinstance(expression, Expr), "Expecting UFL expression.")
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
    
    # check variable cache
    var = variables.get(expression, None)
    if var is not None:
        return var
    
    # skip some types
    skip_classes = (MultiIndex, Zero, FloatValue, IntValue, Identity)
    if isinstance(expression, skip_classes):
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
    # FIXME: Maybe avoid iteration into variables in extract_duplications and handle variables explicitly in here?
    
    # Find all trivially duplicated expressions
    duplications = extract_duplications(expression)
    
    # Mapping: expression -> Variable
    variables = {}
    
    handlers = UFLTypeDict()
    
    # Default transformation handler
    def m_default(x, *ops):
        v = variables.get(x, None)
        if v is None:
            # check if original is in duplications
            in_duplications = (x in duplications)
            # reconstruct if necessary
            if ops != x.operands():
                x = type(x)(*ops)
                # check if reconstructed expression is in duplications (TODO: necessary?)
                in_duplications |= (x in duplications)
            
            # wrap in variable if necessary, or return (possibly reconstructed) expression
            if in_duplications:
                v = Variable(x)
                variables[v._expression] = v
            else:
                v = x
        return v
    for c in ufl_classes:
        handlers[c] = m_default
    
    # Always reuse some constant types
    def m_reuse(x):
        return x
    skip_classes = (MultiIndex, Zero, FloatValue, IntValue, Identity)
    for c in skip_classes:
        handlers[c] = m_reuse
    
    # Recurse differently with variable
    def m_variable(x):
        e = x._expression
        v = variables.get(e, None)
        if v is None:
            e = transform(e, handlers) # FIXME: Will return new variable...
            #if x._expression is e:
            v = Variable(e, x._count)
            variables[e] = v
        return v
    handlers[Variable] = m_variable
    
    # Initialize variable dict with existing variables
    #vars = extract_variables(expression)
    #for v in vars:
    #    variables[v._expression] = v
    
    return transform(expression, handlers)


# TODO: Indices will often mess up extract_duplications / mark_duplications.
# Can we renumber indices consistently from the leaves to avoid that problem?
# This may introduce many ComponentTensor/Indexed objects for relabeling of indices though.
# We probably need some kind of pattern matching to make this effective.
# That's another step towards a complete symbolic library...
# 
# What this does do well is insert Variables around subexpressions that the
# user actually identified manually in his code like in "a = ...; b = a*(1+a)",
# and expressions without indices (prior to expand_compounds).
