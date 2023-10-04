"""Check arities."""
from itertools import chain

from ufl.corealg.traversal import traverse_unique_terminals
from ufl.corealg.multifunction import MultiFunction
from ufl.corealg.map_dag import map_expr_dag
from ufl.classes import Argument, Zero


class ArityMismatch(BaseException):
    """Arity mismatch exception."""
    pass


def _afmt(atuple):
    """Return a string representation of an arity tuple."""
    return tuple(f"conj({arg})" if conj else str(arg) for arg, conj in atuple)


class ArityChecker(MultiFunction):
    """Arity checker."""

    def __init__(self, arguments):
        """Initialise."""
        MultiFunction.__init__(self)
        self.arguments = arguments
        self._et = ()

    def terminal(self, o):
        """Apply to terminal."""
        return self._et

    def argument(self, o):
        """Apply to argument."""
        return ((o, False),)

    def nonlinear_operator(self, o):
        """Apply to nonlinear_operator."""
        # Cutoff traversal by not having *ops in argument list of this
        # handler.  Traverse only the terminals under here the fastest
        # way we know of:
        for t in traverse_unique_terminals(o):
            if t._ufl_typecode_ == Argument._ufl_typecode_:
                raise ArityMismatch(f"Applying nonlinear operator {o._ufl_class_.__name__} to "
                                    f"expression depending on form argument {t}.")
        return self._et

    expr = nonlinear_operator

    def sum(self, o, a, b):
        """Apply to sum."""
        if a != b:
            raise ArityMismatch(f"Adding expressions with non-matching form arguments {_afmt(a)} vs {_afmt(b)}.")
        return a

    def division(self, o, a, b):
        """Apply to division."""
        if b:
            raise ArityMismatch(f"Cannot divide by form argument {b}.")
        return a

    def product(self, o, a, b):
        """Apply to product."""
        if a and b:
            # Check that we don't have test*test, trial*trial, even
            # for different parts in a block system
            anumbers = set(x[0].number() for x in a)
            for x in b:
                if x[0].number() in anumbers:
                    raise ArityMismatch("Multiplying expressions with overlapping form argument number "
                                        f"{x[0].number()}, argument is {_afmt(x)}.")
            # Combine argument lists
            c = tuple(sorted(set(a + b), key=lambda x: (x[0].number(), x[0].part())))
            # Check that we don't have any arguments shared between a
            # and b
            if len(c) != len(a) + len(b) or len(c) != len({x[0] for x in c}):
                raise ArityMismatch("Multiplying expressions with overlapping form arguments "
                                    f"{_afmt(a)} vs {_afmt(b)}.")
            # It's fine for argument parts to overlap
            return c
        elif a:
            return a
        else:
            return b

    # inner, outer and dot all behave as product but for conjugates
    def inner(self, o, a, b):
        """Apply to inner."""
        return self.product(o, a, self.conj(None, b))

    dot = inner

    def outer(self, o, a, b):
        """Apply to outer."""
        return self.product(o, self.conj(None, a), b)

    def linear_operator(self, o, a):
        """Apply to linear_operator."""
        return a

    # Positive and negative restrictions behave as linear operators
    positive_restricted = linear_operator
    negative_restricted = linear_operator

    # Cell and facet average are linear operators
    cell_avg = linear_operator
    facet_avg = linear_operator

    # Grad is a linear operator
    grad = linear_operator
    reference_grad = linear_operator
    reference_value = linear_operator

    # Conj, is a sesquilinear operator
    def conj(self, o, a):
        """Apply to conj."""
        return tuple((a_[0], not a_[1]) for a_ in a)

    # Does it make sense to have a Variable(Argument)? I see no
    # problem.
    def variable(self, o, f, a):
        """Apply to variable."""
        return f

    # Conditional is linear on each side of the condition
    def conditional(self, o, c, a, b):
        """Apply to conditional."""
        if c:
            raise ArityMismatch(f"Condition cannot depend on form arguments ({_afmt(a)}).")
        if a and isinstance(o.ufl_operands[2], Zero):
            # Allow conditional(c, arg, 0)
            return a
        elif b and isinstance(o.ufl_operands[1], Zero):
            # Allow conditional(c, 0, arg)
            return b
        elif a == b:
            # Allow conditional(c, test, test)
            return a
        else:
            # Do not allow e.g. conditional(c, test, trial),
            # conditional(c, test, nonzeroconstant)
            raise ArityMismatch("Conditional subexpressions with non-matching form arguments "
                                f"{_afmt(a)} vs {_afmt(b)}.")

    def linear_indexed_type(self, o, a, i):
        """Apply to linear_indexed_type."""
        return a

    # All of these indexed thingies behave as a linear_indexed_type
    indexed = linear_indexed_type
    index_sum = linear_indexed_type
    component_tensor = linear_indexed_type

    def list_tensor(self, o, *ops):
        """Apply to list_tensor."""
        args = set(chain(*ops))
        if args:
            # Check that each list tensor component has the same
            # argument numbers (ignoring parts)
            numbers = set(tuple(sorted(set(arg[0].number() for arg in op))) for op in ops)
            if () in numbers:  # Allow e.g. <v[0], 0, v[1]> but not <v[0], u[0]>
                numbers.remove(())
            if len(numbers) > 1:
                raise ArityMismatch("Listtensor components must depend on the same argument numbers, "
                                    f"found {numbers}.")

            # Allow different parts with the same number
            return tuple(sorted(args, key=lambda x: (x[0].number(), x[0].part())))
        else:
            # No argument dependencies
            return self._et


def check_integrand_arity(expr, arguments, complex_mode=False):
    """Check the arity of an integrand."""
    arguments = tuple(sorted(set(arguments),
                             key=lambda x: (x.number(), x.part())))
    rules = ArityChecker(arguments)
    arg_tuples = map_expr_dag(rules, expr, compress=False)
    args = tuple(a[0] for a in arg_tuples)
    if args != arguments:
        raise ArityMismatch(f"Integrand arguments {args} differ from form arguments {arguments}.")
    if complex_mode:
        # Check that the test function is conjugated and that any
        # trial function is not conjugated. Further arguments are
        # treated as trial funtions (i.e. no conjugation) but this
        # might not be correct.
        for arg, conj in arg_tuples:
            if arg.number() == 0 and not conj:
                raise ArityMismatch("Failure to conjugate test function in complex Form")
            elif arg.number() > 0 and conj:
                raise ArityMismatch(f"Argument {arg} is spuriously conjugated in complex Form")


def check_form_arity(form, arguments, complex_mode=False):
    """Check the arity of a form."""
    for itg in form.integrals():
        check_integrand_arity(itg.integrand(), arguments, complex_mode)
