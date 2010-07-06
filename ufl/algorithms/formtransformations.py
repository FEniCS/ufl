"""This module defines utilities for transforming
complete Forms into new related Forms."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-10-01"

# Modified by Anders Logg, 2008-2009.
# Modified by Garth N. Wells, 2010.
# Modified by Marie E. Rognes, 2010.

# Last changed: 2010-07-06

from itertools import izip

from ufl.common import some_key, product, Stack
from ufl.log import error, warning
from ufl.assertions import ufl_assert

# All classes:
from ufl.expr import Expr
from ufl.argument import Argument
from ufl.coefficient import Coefficient
from ufl.constantvalue import Zero
from ufl.indexed import Indexed
from ufl.indexsum import IndexSum

# Lists of all Expr classes
from ufl.classes import ufl_classes, terminal_classes, nonterminal_classes

# Other algorithms:
from ufl.algorithms.traversal import traverse_terminals
from ufl.algorithms.analysis import extract_arguments
from ufl.algorithms.transformations import replace, Transformer, apply_transformer, transform_integrands

class PartExtracter(Transformer):
    def __init__(self, arguments):
        Transformer.__init__(self)
        self._want = Stack()
        self._want.push(set(arguments))

    def expr(self, x):
        "The default is a nonlinear operator not accepting any basis functions in its children."
        # TODO: Other operators to implement particularly? Will see when errors here trigger...
        if any(isinstance(t, Argument) for t in traverse_terminals(x)):
            error("Found basis function in %s, this is an invalid expression." % repr(x))
        return (x, set())
    terminal = expr

    def variable(self, o):
        # Check variable cache to reuse previously transformed variable if possible
        e, l = o.operands()
        res = self._variable_cache.get(l)
        if res is not None:
            return res

        # Visit the expression our variable represents
        e2, provides = self.visit(e)

        # If the expression is the same, reuse Variable object
        if e == e2:
            v = o
        else:
            # Strip Variable (expression does not represent the same value here in PartExtracter)
            v = e2

        res = v, provides

        # Cache variable
        self._variable_cache[l] = res
        return res

    def argument(self, x):
        "An argument provides itself, and the requirement can't include any more than itself."
        return (x, set((x,)))

    def sum(self, x):
        "A sum requires nothing of its children, but only reuses those children who provides what is required."
        want = self._want.peek()
        provides = set()

        # Filter operands providing too many basis functions
        ops = []
        for op in x.operands():
            o, o_provides = self.visit(op)
            # if o provides more than we want, skip it
            if not (o_provides - want):
                if len(o_provides) > len(provides):
                    provides = o_provides
                ops.append((o, o_provides))

        # Filter operands providing too few basis functions
        ops2 = []
        for o, o_provides in ops:
            if len(o_provides) == len(provides):
                if o_provides == provides:
                    ops2.append(o)
                else:
                    error("Invalid sum of expressions with incompatible basis function configurations: %s" % repr(x))
            else:
                pass
        from ufl.common import lstr

        # Reuse or reconstruct
        if ops2:
            x = self.reuse_if_possible(x, *ops2)
        else:
            op0 = x.operands()[0]
            x = Zero(op0.shape(), op0.free_indices(), op0.index_dimensions())
        return (x, provides)

    def product(self, x, *ops):
        provides = []
        ops2 = []
        for o, o_provides in ops:

            # Return zero if any factor is an indexed zero
            if isinstance(o, Indexed) and isinstance(o._expression, Zero):
                return (0*x, set([]))

            provides.extend(o_provides)
            ops2.append(o)
        n = len(provides)
        provides = set(provides)
        m = len(provides)
        ufl_assert(m == n, "Found product of basis functions, forms must be linear in each basis function argument: %s" % repr(x))
        x = self.reuse_if_possible(x, *ops2)
        return (x, provides)

    inner = product
    outer = product
    dot = product

    def division(self, x, *ops):
        # FIXME: Check logic of this function

        # Get numerator and denominator
        numerator, denominator = x.operands()

        provides = []

        # Visit numerator
        numerator_x, numerator_provides = self.visit(numerator)
        provides.extend(numerator_provides)

        if isinstance(numerator_x, Zero):
            return (0*x, set([]))

        # Visit denominator
        denominator_x, denominator_provides = self.visit(denominator)
        provides.extend(denominator_provides)

        # Check for basis function in the denominator
        if any(isinstance(t, Argument) for t in traverse_terminals(denominator)):
            error("Found basis function in denominator of %s , this is an invalid expression." % repr(x))

        # FIXME: Should we try using 'reuse_if_possible'?

        provides = set(provides)
        return (x, provides)

    def linear_operator(self, x, arg):
        "A linear operator in a single argument accepting arity > 0, providing whatever basis functions its argument does."

        o, provides = arg
        #x = self.reuse_if_possible(x, (o,)) # commented out, seems to break positive/negative restrictions

        if isinstance(o, Zero):
            return (0*x, set([]))

        return (x, provides)
    # TODO: List all linear operators (use subclassing to simplify stuff like this?)
    positive_restricted = linear_operator
    negative_restricted = linear_operator

    def linear_indexed_type(self, x):

        f, i = x.operands()
        f2, provides = self.visit(f)
        x = self.reuse_if_possible(x, f2, i)

        if isinstance(x, Zero):
            return (0*x, set([]))

        return (x, provides)

    def indexed(self, x):

        f, i = x.operands()
        f2, provides = self.visit(f)
        x = self.reuse_if_possible(x, f2, i)

        if isinstance(x._expression, Zero):
            return (0*x, set([]))

        return (x, provides)

    index_sum = linear_indexed_type
    component_tensor = linear_indexed_type
    spatial_derivative = linear_indexed_type

    def list_tensor(self, x, *ops):
        provides = ops[0][1]
        oprov = [o[1] for o in ops]
        ops2  = [o[0] for o in ops]
        s = "\n\n".join("%s\n%s" % (a,b) for (a,b) in ops)
        #ufl_assert(all(provides == op for op in oprov),\
        #        "List tensor elements provide different properties, invalid expression.\n%s" % (s,))
        ufl_assert(all(isinstance(o, Expr) for o in ops2), \
                "Got wrong types in list_tensor handler.")
        x = self.reuse_if_possible(x, *ops2)
        return (x, provides)

def compute_form_with_arity(form, arity): # TODO: Test and finish
    """Compute the left hand side of a form."""

    arguments = extract_arguments(form)

    if len(arguments) < arity:
        warning("Form has no parts with arity %d." % arity)
        return 0*form

    arguments = set(arguments[:arity])
    pe = PartExtracter(arguments)
    def _transform(e):
        e, provides = pe.visit(e)
        if provides == arguments:
            return e
        return Zero()
    res = transform_integrands(form, _transform)
    return res

def compute_form_arities(form):
    """Return set of arities of terms present in form."""

    def _transform(e):
        e, provides = pe.visit(e)
        if provides == sub_arguments:
            return e
        return Zero()

    arities = set()
    arguments = extract_arguments(form)

    for arity in range(len(arguments)+1):
        sub_arguments = set(arguments[:arity])
        pe = PartExtracter(sub_arguments)
        for itg in form.integrals():
            integrand = _transform(itg.integrand())
            if not integrand or isinstance(integrand, Zero):
                continue
            arities.add(arity)
            break

    return arities

def compute_form_lhs(form):
    """Compute the left hand side of a form.

    Example:

        a = u*v*dx + f*v*dx
        a = lhs(a) -> u*v*dx
    """
    return compute_form_with_arity(form, 2)

def compute_form_rhs(form):
    """Compute the right hand side of a form.

    Example:

        a = u*v*dx + f*v*dx
        L = rhs(a) -> -f*v*dx
    """
    return -compute_form_with_arity(form, 1)

def compute_form_functional(form):
    """Compute the functional part of a form, that
    is the terms independent of basis functions.

    (Used for testing, not sure if it's useful for anything?)"""
    return compute_form_with_arity(form, 0)

def compute_form_action(form, function):
    """Compute the action of a form on a Coefficient.

    This works simply by replacing the last basis_function
    with a Coefficient on the same function space (element).
    The form returned will thus have one Argument less
    and one additional Coefficient at the end if no function
    has been provided.
    """
    arguments = extract_arguments(form)
    if len(arguments) == 2:
        v, u = arguments
    elif len(arguments) == 1:
        u, = arguments
    else:
        error("Expecting bilinear or linear form.")

    e = u.element()
    if function is None:
        function = Coefficient(e)
    else:
        ufl_assert(function.element() == e, \
            "Trying to compute action of form on a "\
            "function in an incompatible element space.")
    return replace(form, { u: function })

def compute_energy_norm(form, function):
    """Compute the a-norm of a Coefficient given a form a.

    This works simply by replacing the two basis functions
    with a Coefficient on the same function space (element).
    The Form returned will thus be a functional with no
    basis functions, and one additional Coefficient at the
    end if no function has been provided.
    """
    arguments = extract_arguments(form)
    ufl_assert(len(arguments) == 2, "Expecting bilinear form.")
    v, u = arguments
    e = u.element()
    e2 = v.element()
    ufl_assert(e == e2, "Expecting equal finite elements for test and trial functions, got '%s' and '%s'." % (str(e), str(e2)))
    if function is None:
        function = Coefficient(e)
    else:
        ufl_assert(function.element() == e, \
            "Trying to compute action of form on a "\
            "function in an incompatible element space.")
    return replace(form, { u: function, v: function })

def compute_form_adjoint(form):
    """Compute the adjoint of a bilinear form.

    This works simply by swapping the first and last arguments.
    """
    arguments = extract_arguments(form)
    ufl_assert(len(arguments) == 2, "Expecting bilinear form.")
    v, u = arguments
    return replace(form, {v: u, u: v})

#def compute_dirichlet_functional(form):
#    """Compute the Dirichlet functional of a form:
#    a(v,u;...) - L(v; ...) -> 0.5 a(v,v;...) - L(v;...)
#
#    This assumes a bilinear form and works simply by
#    replacing the trial function with the test function.
#    The form returned will thus be a linear form.
#    """
#    warning("TODO: Don't know if this is correct or even useful, just picked up the name some place.")
#    return 0.5*compute_form_lhs(form) - compute_form_rhs(form)
#    #bf = extract_arguments(form)
#    #ufl_assert(len(bf) == 2, "Expecting bilinear form.")
#    #v, u = bf
#    #return replace(form, {u:v})
