"""Utilities for transforming complete Forms into new related Forms."""

# Copyright (C) 2008-2025 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2008-2009.
# Modified by Garth N. Wells, 2010.
# Modified by Marie E. Rognes, 2010.
# Modified by Jørgen S. Dokken, 2025.

import logging

from ufl.algebra import Conj
from ufl.algorithms.formsplitter import extract_blocks

# Other algorithms:
from ufl.algorithms.map_integrands import map_integrands
from ufl.algorithms.replace import replace
from ufl.algorithms.transformer import Transformer
from ufl.argument import Argument
from ufl.coefficient import Coefficient
from ufl.constantvalue import Zero

# All classes:
from ufl.core.expr import ufl_err_str

logger = logging.getLogger("ufl")


# FIXME: Don't use this below, it makes partextracter more expensive than necessary
def _expr_has_terminal_types(expr, ufl_types):
    """Check if an expression has terminal types."""
    input = [expr]
    while input:
        e = input.pop()
        ops = e.ufl_operands
        if ops:
            input.extend(ops)
        elif isinstance(e, ufl_types):
            return True
    return False


def zero_expr(e):
    """Create a zero expression."""
    return Zero(e.ufl_shape, e.ufl_free_indices, e.ufl_index_dimensions)


class PartExtracter(Transformer):
    """PartExtracter extracts those parts of a form that contain the given argument(s)."""

    def __init__(self, arguments):
        """Initialise."""
        Transformer.__init__(self)
        self._want = set(arguments)

    def expr(self, x):
        """Apply to expr.

        The default is a nonlinear operator not accepting any Arguments among its children.
        """
        if _expr_has_terminal_types(x, Argument):
            raise ValueError(f"Found Argument in {ufl_err_str(x)}, this is an invalid expression.")
        return (x, set())

    # Terminals that are not Variables or Arguments behave as default
    # expr-s.
    terminal = expr

    def variable(self, x):
        """Return relevant parts of this variable."""
        # Extract parts/provides from this variable's expression
        expression, label = x.ufl_operands
        part, provides = self.visit(expression)

        # If the extracted part is zero or we provide more than we
        # want, return zero
        if isinstance(part, Zero) or (provides - self._want):
            return (zero_expr(x), set())

        # Reuse varible if possible (or reconstruct from part)
        x = self.reuse_if_possible(x, part, label)

        return (x, provides)

    def argument(self, x):
        """Return itself unless itself provides too much."""
        # An argument provides itself
        provides = {x}

        # If we provide more than we want, return zero
        if provides - self._want:
            return (zero_expr(x), set())

        return (x, provides)

    def sum(self, x):
        """Return the terms that might eventually yield the correct parts(!).

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
        error (e.g. Argument(-1) + Argument(-2)).
        """
        parts_that_provide = {}

        # 1. Skip terms that provide too much
        original_terms = x.ufl_operands
        assert len(original_terms) == 2
        for term in original_terms:
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
            return (zero_expr(x), set())

        # 3. Return the terms that provide the biggest set
        most_provided = frozenset()
        for provideds, parts in parts_that_provide.items():  # TODO: Just sort instead?
            # Throw error if size of sets are equal (and not zero)
            if len(provideds) == len(most_provided) and len(most_provided):
                raise ValueError("Don't know what to do with sums with different Arguments.")

            if provideds > most_provided:
                most_provided = provideds

        terms = parts_that_provide[most_provided]
        if len(terms) == 2:
            x = self.reuse_if_possible(x, *terms)
        else:
            (x,) = terms

        return (x, most_provided)

    def product(self, x, *ops):
        """Apply to product.

        Note: Product is a visit-children-first handler. ops are
        the visited factors.
        """
        provides = set()
        factors = []

        for factor, factor_provides in ops:
            # If any factor is zero, return
            if isinstance(factor, Zero):
                return (zero_expr(x), set())

            # Add factor to factors and extend provides
            factors.append(factor)
            provides = provides | factor_provides

            # If we provide more than we want, return zero
            if provides - self._want:
                return (zero_expr(x), provides)

        # Reuse product if possible (or reconstruct from factors)
        x = self.reuse_if_possible(x, *factors)

        return (x, provides)

    # inner, outer and dot all behave as product
    inner = product
    outer = product
    dot = product

    def division(self, x):
        """Return parts_of_numerator/denominator."""
        # Get numerator and denominator
        numerator, denominator = x.ufl_operands

        # Check for Arguments in the denominator
        if _expr_has_terminal_types(denominator, Argument):
            raise ValueError(
                f"Found Argument in denominator of {ufl_err_str(x)}, this is an invalid expression."
            )

        # Visit numerator
        numerator_parts, provides = self.visit(numerator)

        # If numerator is zero, return zero. (No need to check whether
        # it provides too much, already checked by visit.)
        if isinstance(numerator_parts, Zero):
            return (zero_expr(x), set())

        # Reuse x if possible, otherwise reconstruct from (parts of)
        # numerator and denominator
        x = self.reuse_if_possible(x, numerator_parts, denominator)

        return (x, provides)

    def linear_operator(self, x, arg):
        """Apply to linear_operator.

        A linear operator with a single operand accepting arity > 0,
        providing whatever Argument its operand does.
        """
        # linear_operator is a visit-children-first handler. Handled
        # arguments are in arg.
        part, provides = arg

        # Discard if part is zero. (No need to check whether we
        # provide too much, already checked by children.)
        if isinstance(part, Zero):
            return (zero_expr(x), set())

        x = self.reuse_if_possible(x, part)

        return (x, provides)

    # Positive and negative restrictions behave as linear operators
    positive_restricted = linear_operator
    negative_restricted = linear_operator

    # Cell and facet average are linear operators
    cell_avg = linear_operator
    facet_avg = linear_operator

    # Grad is a linear operator
    grad = linear_operator

    # Conj, Real, Imag are linear operators
    conj = linear_operator
    real = linear_operator
    imag = linear_operator

    def linear_indexed_type(self, x):
        """Return parts of expression belonging to this indexed expression."""
        expression, index = x.ufl_operands
        part, provides = self.visit(expression)

        # Return zero if extracted part is zero. (The expression
        # should already have checked if it provides too much.)
        if isinstance(part, Zero):
            return (zero_expr(x), set())

        # Reuse x if possible (or reconstruct by indexing part)
        x = self.reuse_if_possible(x, part, index)

        return (x, provides)

    # All of these indexed thingies behave as a linear_indexed_type
    indexed = linear_indexed_type
    index_sum = linear_indexed_type
    component_tensor = linear_indexed_type

    def list_tensor(self, x, *ops):
        """Apply to list_tensor."""
        # list_tensor is a visit-children-first handler. ops contains
        # the visited operands with their provides. (It follows that
        # none of the visited operands provide more than wanted.)

        # Extract the most arguments provided by any of the components
        most_provides = ops[0][1]
        for component, provides in ops:
            if provides - most_provides:
                most_provides = provides

        # Check that all components either provide the same arguments
        # or vanish. (This check is here b/c it is not obvious what to
        # return if the components provide different arguments, at
        # least with the current transformer design.)
        for component, provides in ops:
            if provides != most_provides and not isinstance(component, Zero):
                raise ValueError(
                    "PartExtracter does not know how to handle list_tensors with "
                    "non-zero components providing fewer arguments"
                )

        # Return components
        components = [op[0] for op in ops]
        x = self.reuse_if_possible(x, *components)

        return (x, most_provides)


def compute_form_with_arity(form, arity, arguments=None):
    """Compute parts of form of given arity.

    Note:
        If the form has no parts of the specified arity, this will return 0*form.
    """
    # Extract all arguments in form
    if arguments is None:
        arguments = form.arguments()

    if len(arguments) < arity:
        return 0 * form

    # Assuming that the form is not a sum of terms
    # that depend on different arguments, e.g. (u+v)*dx
    # would result in just v*dx. But that doesn't make
    # any sense anyway.
    sub_arguments = set(arguments[:arity])
    pe = PartExtracter(sub_arguments)

    def _transform(e):
        e_visited, provides = pe.visit(e)
        if provides == sub_arguments:
            return e_visited
        return Zero()

    return map_integrands(_transform, form)


def compute_form_arities(form):
    """Return set of arities of terms present in form."""
    # Extract all arguments present in form
    arguments = form.arguments()

    parts = [arg.part() for arg in arguments]
    if set(parts) - {None}:
        raise ValueError("compute_form_arities cannot handle parts.")

    arities = set()
    for arity in range(len(arguments) + 1):
        # Compute parts with arity "arity"
        parts = compute_form_with_arity(form, arity, arguments)

        # Register arity if "parts" does not vanish
        if parts and parts.integrals():
            arities.add(arity)

    return arities


def compute_form_lhs(form):
    """Compute the left hand side of a form.

    Example:
        a = u*v*dx + f*v*dx
        a = lhs(a) -> u*v*dx
    """
    parts = tuple(sorted(set(part for a in form.arguments() if (part := a.part()) is not None)))
    # If Arguments are stemming from a MixedFunctionSpace, we have to compute this per block
    if parts == ():
        return compute_form_with_arity(form, 2)

    form_blocks = extract_blocks(form, arity=2)
    lhs = 0
    for bi in form_blocks:
        for bj in bi:
            if bj is not None:
                lhs += compute_form_with_arity(bj, 2)
    return lhs


def compute_form_rhs(form):
    """Compute the right hand side of a form.

    Example:
        a = u*v*dx + f*v*dx
        L = rhs(a) -> -f*v*dx
    """
    parts = tuple(sorted(set(part for a in form.arguments() if (part := a.part()) is not None)))
    if parts == ():
        return -compute_form_with_arity(form, 1)

    form_blocks = extract_blocks(form, arity=1)
    rhs = 0
    for bi in form_blocks:
        if bi is not None:
            rhs += compute_form_with_arity(bi, 1)
    return -rhs


def compute_form_functional(form):
    """Compute the functional part of a form, that is the terms independent of Arguments.

    (Used for testing, not sure if it's useful for anything?)
    """
    return compute_form_with_arity(form, 0)


def compute_form_action(form, coefficient):
    """Compute the action of a form on a Coefficient.

    This works simply by replacing the last Argument
    with a Coefficient on the same function space (element).
    The form returned will thus have one Argument less
    and one additional Coefficient at the end if no
    Coefficient has been provided.
    """
    # TODO: Check whatever makes sense for coefficient

    # Extract all arguments
    arguments = form.arguments()

    parts = [arg.part() for arg in arguments]
    if set(parts) - {None}:
        # We assume that for MixedFunctionSpace that the max arity can be two
        highest_arity_form = compute_form_lhs(form)
        if highest_arity_form == 0 or highest_arity_form.empty():
            highest_arity_form = compute_form_rhs(form)
            if highest_arity_form == 0 or highest_arity_form.empty():
                raise ValueError("No arguments to replace in form.")
        arguments = highest_arity_form.arguments()
        numbers = [a.number() for a in arguments]
        max_number = max(numbers)
        if coefficient is None:
            replacement_map = {
                a: Coefficient(a.ufl_function_space())
                for a in arguments
                if a.number() == max_number
            }
        else:
            replacement_map = {
                a: coefficient[a.part()] for a in arguments if a.number() == max_number
            }
        return replace(form, replacement_map)

    # Pick last argument (will be replaced)
    u = arguments[-1]

    fs = u.ufl_function_space()
    if coefficient is None:
        coefficient = Coefficient(fs)
    elif coefficient.ufl_function_space() != fs:
        logger.debug("Computing action of form on a coefficient in a different function space.")
    return replace(form, {u: coefficient})


def compute_energy_norm(form, coefficient):
    """Compute the a-norm of a Coefficient given a form a.

    This works simply by replacing the two Arguments
    with a Coefficient on the same function space (element).
    The Form returned will thus be a functional with no
    Arguments, and one additional Coefficient at the
    end if no coefficient has been provided.
    """
    from ufl.formoperators import action  # Delayed import to avoid circularity

    arguments = form.arguments()

    parts = [arg.part() for arg in arguments]
    if set(parts) - {None}:
        raise ValueError("compute_energy_norm cannot handle parts.")

    if len(arguments) != 2:
        raise ValueError("Expecting bilinear form.")
    v, u = arguments
    U = u.ufl_function_space()
    V = v.ufl_function_space()
    if U != V:
        raise ValueError(
            f"Expecting equal finite elements for test and trial functions, got '{U}' and '{V}'."
        )
    if coefficient is None:
        coefficient = Coefficient(V)
    else:
        if coefficient.ufl_function_space() != U:
            raise ValueError(
                "Trying to compute action of form on a "
                "coefficient in an incompatible element space."
            )
    return action(action(form, coefficient), coefficient)


def compute_form_adjoint(form, reordered_arguments=None):
    """Compute the adjoint of a bilinear form.

    This works simply by swapping the number and part of the two arguments,
    but keeping their elements and places in the integrand expressions.
    """
    arguments = form.arguments()
    if form.empty():
        return form
    parts = [arg.part() for arg in arguments]
    if set(parts) - {None}:
        J = extract_blocks(form, arity=2)
        num_blocks = len(J)
        J_adj = 0
        for i in range(num_blocks):
            for j in range(num_blocks):
                if J[i][j] is None:
                    continue
                v, u = J[i][j].arguments()
                if reordered_arguments is None:
                    reordered_u = Argument(u.ufl_function_space(), number=v.number(), part=v.part())
                    reordered_v = Argument(v.ufl_function_space(), number=u.number(), part=u.part())
                else:
                    reordered_u, reordered_v = reordered_arguments[i]

                if reordered_u.number() >= reordered_v.number():
                    raise ValueError("Ordering of new arguments is the same as the old arguments!")

                if reordered_u.part() != v.part():
                    raise ValueError("Ordering of new arguments is the same as the old arguments!")
                if reordered_v.part() != u.part():
                    raise ValueError("Ordering of new arguments is the same as the old arguments!")

                if reordered_u.ufl_function_space() != u.ufl_function_space():
                    raise ValueError(
                        "Element mismatch between new and old arguments (trial functions)."
                    )
                if reordered_v.ufl_function_space() != v.ufl_function_space():
                    raise ValueError(
                        "Element mismatch between new and old arguments (test functions)."
                    )

                J_adj += map_integrands(Conj, replace(J[i][j], {v: reordered_v, u: reordered_u}))
        return J_adj

    if len(arguments) != 2:
        raise ValueError("Expecting bilinear form.")

    v, u = arguments
    if v.number() >= u.number():
        raise ValueError("Mistaken assumption in code!")

    if reordered_arguments is None:
        reordered_u = Argument(u.ufl_function_space(), number=v.number(), part=v.part())
        reordered_v = Argument(v.ufl_function_space(), number=u.number(), part=u.part())
    else:
        reordered_u, reordered_v = reordered_arguments

    if reordered_u.number() >= reordered_v.number():
        raise ValueError("Ordering of new arguments is the same as the old arguments!")

    if reordered_u.part() != v.part():
        raise ValueError("Ordering of new arguments is the same as the old arguments!")
    if reordered_v.part() != u.part():
        raise ValueError("Ordering of new arguments is the same as the old arguments!")

    if reordered_u.ufl_function_space() != u.ufl_function_space():
        raise ValueError("Element mismatch between new and old arguments (trial functions).")
    if reordered_v.ufl_function_space() != v.ufl_function_space():
        raise ValueError("Element mismatch between new and old arguments (test functions).")

    return map_integrands(Conj, replace(form, {v: reordered_v, u: reordered_u}))
