from ufl import *

### Traversal utilities

def iter_depth_first(u):
    for o in u.ops():
        for i in iter_depth_first(o):
            yield i
    yield u

def iter_width_first(u):
    yield u
    for o in u.ops():
        for i in iter_width_first(o):
            yield i

def traverse_depth_first(func, u):
    for o in u.ops():
        traverse_depth_first(func, o)
    func(u)

def traverse_width_first(func, u):
    for o in u.ops():
        traverse_depth_first(func, o)
    func(u)


### Utilities for iteration over particular types

def ufl_objs(u):
    """Utility function to handle Form, Integral and any UFLObject the same way."""
    if isinstance(u, Form):
        objs = (itg.integrand for itg in u.integrals)
    elif isinstance(u, Integral):
        objs = (u.integrand,)
    else:
        objs = (u,)
    return objs

def iter_classes(u):
    """Returns an iterator over the unique classes used by objects in this expression."""
    returned = set()
    for o in ufl_objs(u):
        for u in iter_depth_first(o):
            t = u.__class__
            if not t in returned:
                returned.add(t)
                yield t

def iter_elements(u):
    """Returns an iterator over the unique finite elements used in this form or expression."""
    returned = set()
    for o in ufl_objs(u):
        for u in iter_depth_first(o):
            if isinstance(u, (BasisFunction, UFLCoefficient)) and not repr(u.element) in returned:
                returned.add(repr(u.element))
                yield u.element

def iter_basisfunctions(u):
    """Returns an iterator over the unique basis functions used in this form or expression."""
    returned = set()
    for o in ufl_objs(u):
        for u in iter_depth_first(o):
            if isinstance(u, BasisFunction) and not repr(u) in returned:
                returned.add(repr(u))
                yield u

def iter_coefficients(u):
    """Returns an iterator over the unique coefficient functions used in this form or expression."""
    returned = set()
    for o in ufl_objs(u):
        for u in iter_depth_first(o):
            if isinstance(u, UFLCoefficient) and not repr(u) in returned:
                returned.add(repr(u))
                yield u


### Utilities to convert expression to a different form

def flatten_sums_and_products(u):
    # FIXME: how should we do this? visitor pattern? avoid bloating... need general way to parse an expression and replace subtrees by new expressions
    ops = []
    for o in u.ops():
        ops.append( flatten_sums_and_products(o) )
    if isinstance(u, Sum):
        pass # FIXME
    elif isinstance(u, Product):
        pass # FIXME
    return u.__class__(*ops) # FIXME: this is assuming ops and __init__ are matching, which they in general aren't

