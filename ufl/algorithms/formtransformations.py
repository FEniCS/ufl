"""This module defines utilities for transforming
complete Forms into new related Forms."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-10-01"

# Modified by Anders Logg, 2008-2009.
# Modified by Garth N. Wells, 2010.
# Modified by Marie E. Rognes, 2010.

# Last changed: 2010-10-25

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
from ufl.algorithms.preprocess import preprocess
from ufl.algorithms.transformations import replace, Transformer, apply_transformer, transform_integrands

def zero(e):
    return Zero(e.shape(), e.free_indices(), e.index_dimensions())

class PartExtracter(Transformer):
    """
    PartExtracter extracts those parts of a form that contain the
    given argument(s).
    """

    def __init__(self, arguments):
        Transformer.__init__(self)
        self._want = set(arguments)

    def expr(self, x):
        """The default is a nonlinear operator not accepting any basis
        functions in its children."""

        if any(isinstance(t, Argument) for t in traverse_terminals(x)):
            error("Found basis function in %s, this is an invalid expression." % repr(x))
        return (x, set())

    # Terminals that are not Variables or Arguments behave as default
    # expr-s.
    terminal = expr

    def variable(self, x):
        "Return relevant parts of this variable."

        # Extract parts/provides from this variable's expression
        expression, label = x.operands()
        part, provides = self.visit(expression)

        # If the extracted part is zero or we provide more than we
        # want, return zero
        if isinstance(part, Zero) or (provides - self._want):
            return (zero(x), set())

        # Reuse varible if possible (or reconstruct from part)
        x = self.reuse_if_possible(x, part, label)

        return (x, provides)

    def argument(self, x):
        "Return itself unless itself provides too much."

        # An argument provides itself
        provides = set((x,))

        # If we provide more than we want, return zero
        if provides - self._want:
            return (zero(x), set())

        return (x, provides)

    def sum(self, x):
        """
        Return the terms that might eventually yield the correct
        parts(!)

        The logic required for sums is a bit elaborate:

        A sum may contain terms providing different arguments. We
        should return (a sum of) a suitable subset of these
        terms. Those should all provide the same arguments.

        For each term in a sum, there are 2 simple possibilities:

        1a) The relevant part of the term is zero -> skip.
        1b) The term provides more arguments than we want -> skip

        2) If all terms fall into the above category, we can just
        return zero.

        Any remaining terms may provide exactly the arguments we want,
        or fewer. This is where things start getting interesting.

        3) Bottom-line: if there are terms with providing different
        arguments -- provide terms that contain the most arguments. If
        there are terms providing different sets of same size -> throw
        error (e.g. Argument(-1) + Argument(-2))
        """

        parts_that_provide = {}

        # 1. Skip terms that provide too much
        for term in x.operands():

            # Visit this term in the sum
            part, term_provides = self.visit(term)

            # If this part is zero or it provides more than we want,
            # skip it
            if isinstance(part, Zero) or (term_provides - self._want):
                continue

            # Add the contributions from this part to temporary list
            term_provides = frozenset(term_provides)
            if term_provides in parts_that_provide:
                parts_that_provide[term_provides] += [part]
            else:
                parts_that_provide[term_provides] = [part]

        # 2. If there are no remaining terms, return zero
        if not parts_that_provide:
            return (zero(x), set())

        # 3. Return the terms that provide the biggest set
        most_provided = frozenset()
        for (provideds, parts) in parts_that_provide.iteritems():

            # Throw error if size of sets are equal (and not zero)
            if len(provideds) == len(most_provided) and len(most_provided):
                error("Don't know what to do with sums with different basis functions")

            if provideds > most_provided:
                most_provided = provideds

        terms = parts_that_provide[most_provided]
        x = self.reuse_if_possible(x, *terms)
        return (x, most_provided)


    def product(self, x, *ops):
        """ Note: Product is a visit-children-first handler. ops are
        the visited factors."""

        provides = set()
        factors = []

        for factor, factor_provides in ops:

            # If any factor is zero, return
            if isinstance(factor, Zero):
                return (zero(x), set())

            # Add factor to factors and extend provides
            factors.append(factor)
            provides = provides | factor_provides

            # If we provide more than we want, return zero
            if provides - self._want:
                return (zero(x), provides)

        # Reuse product if possible (or reconstruct from factors)
        x = self.reuse_if_possible(x, *factors)

        return (x, provides)

    # inner, outer and dot all behave as product
    inner = product
    outer = product
    dot = product

    def division(self, x):
        "Return parts_of_numerator/denominator."

        # Get numerator and denominator
        numerator, denominator = x.operands()

        # Check for basis functions in the denominator
        if any(isinstance(t, Argument) for t in traverse_terminals(denominator)):
            error("Found basis function in denominator of %s , this is an invalid expression." % repr(x))

        # Visit numerator
        numerator_parts, provides = self.visit(numerator)

        # If numerator is zero, return zero. (No need to check whether
        # it provides too much, already checked by visit.)
        if isinstance(numerator_parts, Zero):
            return (zero(x), set())

        # Reuse x if possible, otherwise reconstruct from (parts of)
        # numerator and denominator
        x = self.reuse_if_possible(x, numerator_parts, denominator)

        return (x, provides)

    def linear_operator(self, x, arg):
        """A linear operator in a single argument accepting arity > 0,
        providing whatever basis functions its argument does."""

        # linear_operator is a visit-children-first handler. Handled
        # arguments are in arg.
        part, provides = arg

        # Discard if part is zero. (No need to check whether we
        # provide too much, already checked by children.)
        if isinstance(part, Zero):
            return (zero(x), set())

        x = self.reuse_if_possible(x, part)

        return (part, provides)

    # Positive and negative restrictions behave as linear operators
    positive_restricted = linear_operator
    negative_restricted = linear_operator

    def linear_indexed_type(self, x):
        """Return parts of expression belonging to this indexed
        expression."""

        expression, index = x.operands()
        part, provides = self.visit(expression)

        # Return zero if extracted part is zero. (The expression
        # should already have checked if it provides too much.)
        if isinstance(part, Zero):
            return (zero(x), set())

        # Reuse x if possible (or reconstruct by indexing part)
        x = self.reuse_if_possible(x, part, index)

        return (x, provides)

    # All of these indexed thingies behave as a linear_indexed_type
    indexed = linear_indexed_type
    index_sum = linear_indexed_type
    component_tensor = linear_indexed_type
    spatial_derivative = linear_indexed_type

    def list_tensor(self, x, *ops):

        # list_tensor is a visit-children-first handler. ops contains
        # the visited operands with their provides. Check that all
        # provide the same arguments and return all

        default = ops[0][1]
        for (items, provides) in ops:
            if provides != default:
                error("All components of a list tensor most provide same arguments")

        parts = [o[0] for o in ops]

        x = self.reuse_if_possible(x, *parts)

        return (x, default)

def compute_form_with_arity(form, arity):
    """Compute parts of form of given arity."""

    # Preprocess form (preprocess takes care of checking)
    form = preprocess(form)

    # Extract all arguments in form
    arguments = extract_arguments(form)

    if len(arguments) < arity:
        warning("Form has no parts with arity %d." % arity)
        return 0*form

    # FIXME: Should be permutations of arguments of length arity
    sub_arguments = set(arguments[:arity])
    pe = PartExtracter(sub_arguments)
    def _transform(e):
        e, provides = pe.visit(e)
        if provides == sub_arguments:
            return e
        return Zero()
    res = transform_integrands(form, _transform)
    return res

def compute_form_arities(form):
    """Return set of arities of terms present in form."""

    # Extract all arguments present in form
    arguments = extract_arguments(form)
    arities = set()
    for arity in range(len(arguments)+1):

        # Compute parts with arity "arity"
        parts = compute_form_with_arity(form, arity)

        # Register arity if "parts" does not vanish
        if parts and parts._integrals:
            arities.add(arity)

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
        #ufl_assert(function.element() == e, \
        if function.element() != e:
            print "Computing action of form on a " \
                  "function in a different element space."
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

