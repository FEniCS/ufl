"""
Stuff in this file will probably be removed.
"""


from ufl.log import error, warning
from ufl.algorithms.transformer import ReuseTransformer, apply_transformer
from ufl.classes import Variable


class TreeFlattener(ReuseTransformer):
    def __init__(self):
        ReuseTransformer.__init__(self)

    def sum_or_product(self, o, *ops):
        c = o._uflclass
        operands = []
        for b in ops:
            if isinstance(b, c):
                operands.extend(b.operands())
            else:
                operands.append(b)
        return o.reconstruct(*operands)
    sum = sum_or_product
    product = sum_or_product

def flatten(e): # TODO: Fix or remove! Maybe this works better now with IndexSum marking implicit summations.
    """Convert an UFL expression to a new UFL expression, with sums
    and products flattened from binary tree nodes to n-ary tree nodes."""
    warning("flatten doesn't work correctly for some indexed products, like (u[i]*v[i])*(q[i]*r[i])")
    return apply_transformer(e, TreeFlattener())


#class OperatorApplier(ReuseTransformer):
#    "Implements mappings that can be defined through Python operators."
#    def __init__(self):
#        ReuseTransformer.__init__(self)
#
#    def abs(self, o, a):
#        return abs(a)
#
#    def sum(self, o, *ops):
#        return sum(ops)
#
#    def division(self, o, a, b):
#        return a / b
#
#    def power(self, o, a, b):
#        return a ** b
#
#    def product(self, o, *ops):
#        return product(ops)
#
#    def indexed(self, o, a, b):
#        return a[*b] if isinstance(b, tuple) else a[b]


# TODO: Indices will often mess up extract_duplications / mark_duplications.
# Can we renumber indices consistently from the leaves to avoid that problem?
# This may introduce many ComponentTensor/Indexed objects for relabeling of indices though.
# We probably need some kind of pattern matching to make this effective.
# That's another step towards a complete symbolic library...
#
# What this does do well is insert Variables around subexpressions that the
# user actually identified manually in his code like in "a = ...; b = a*(1+a)",
# and expressions without indices (prior to expand_compounds).
class DuplicationMarker(ReuseTransformer):
    def __init__(self, duplications):
        ReuseTransformer.__init__(self)
        self._duplications = duplications
        self._expr2variable = {}

    def expr(self, o, *ops):
        v = self._expr2variable.get(o)
        if v is None:
            oo = o
            # reconstruct if necessary
            if not ops == o.operands():
                o = o._uflclass(*ops)

            if (oo in self._duplications) or (o in self._duplications):
                v = Variable(o)
                self._expr2variable[o] = v
                self._expr2variable[oo] = v
            else:
                v = o
        return v

    def wrap_terminal(self, o):
        v = self._expr2variable.get(o)
        if v is None:
            if o in self._duplications:
                v = Variable(o)
                self._expr2variable[o] = v
            else:
                v = o
        return v
    argument = wrap_terminal
    coefficient = wrap_terminal
    constant = wrap_terminal
    facet_normal = wrap_terminal

    def variable(self, o):
        e, l = o.operands()
        v = self._expr2variable.get(e)
        if v is None:
            e2 = self.visit(e)
            # Unwrap expression from the newly created Variable wrapper
            # unless the original expression was a Variable, in which
            # case we possibly need to keep the label for correctness.
            if (not isinstance(e, Variable)) and isinstance(e2, Variable):
                e2 = e2._expression
            v = self._expr2variable.get(e2)
            if v is None:
                v = Variable(e2, l)
                self._expr2variable[e] = v
                self._expr2variable[e2] = v
        return v

from ufl.algorithms.analysis import extract_duplications
def mark_duplications(e):
    """Wrap subexpressions that are equal
    (completely equal, not mathematically equivalent)
    in Variable objects to facilitate subexpression reuse."""
    duplications = extract_duplications(e)
    return apply_transformer(e, DuplicationMarker(duplications))


class DuplicationPurger(ReuseTransformer):
    "Replace all duplicated nodes from an UFL Expr."
    def __init__(self):
        ReuseTransformer.__init__(self)
        self._handled = {}
        #self._duplications = set()

    def expr(self, x, *ops):
        # Check cache
        e = self._handled.get(x)
        if e is None:
            # Reuse or reconstruct
            if ops == x.operands():
                e = x
            else:
                e = x._uflclass(*ops)
            # Update cache
            self._handled[x] = e
        #else:
        #    self._duplications.add(e)
        assert repr(x) == repr(e)
        return e

    def terminal(self, x):
        e = self._handled.get(x)
        if e is None:
            # Reuse
            e = x
            # Update cache
            self._handled[x] = e
        #else:
        #    self._duplications.add(e)
        return e

def purge_duplications(e):
    """Replace any subexpressions in expression that
    occur more than once with a single instance."""
    return apply_transformer(e, DuplicationPurger())
