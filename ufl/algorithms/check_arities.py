"""Check arities."""

from functools import singledispatchmethod
from itertools import chain

from ufl.classes import (
    Argument,
    CellAvg,
    ComponentTensor,
    Conditional,
    Conj,
    Curl,
    Div,
    Division,
    Dot,
    Expr,
    FacetAvg,
    Grad,
    Indexed,
    IndexSum,
    Inner,
    Interpolate,
    ListTensor,
    NablaDiv,
    NablaGrad,
    NegativeRestricted,
    Outer,
    PositiveRestricted,
    Product,
    ReferenceCurl,
    ReferenceDiv,
    ReferenceGrad,
    ReferenceValue,
    Sum,
    Terminal,
    Variable,
    Zero,
)
from ufl.corealg.dag_traverser import DAGTraverser
from ufl.corealg.traversal import traverse_unique_terminals


class ArityMismatch(BaseException):
    """Arity mismatch exception."""

    pass


def _afmt(atuple: tuple[Argument, bool]) -> str:
    """Return a string representation of an arity tuple."""
    arg, conj = atuple
    return f"conj({arg})" if conj else str(arg)


class ArityChecker(DAGTraverser):
    """Check which form arguments an expression depends on.

    Each processed expression returns a tuple of ``(argument, is_conjugated)``
    pairs.  ``is_conjugated`` records whether the corresponding argument is
    conjugated in the expression.
    """

    def __init__(self, arguments):
        """Initialise."""
        super().__init__(compress=False)
        self.arguments = arguments
        self._et = ()

    @singledispatchmethod
    def process(self, o, **kwargs):
        """Apply to an expression node."""
        return super().process(o, **kwargs)

    @process.register(Expr)
    def nonlinear_operator(self, o, **kwargs):
        """Apply to a nonlinear operator."""
        # Do not traverse children here. Traverse only the terminals under
        # this node to retain the cutoff behaviour of the old MultiFunction.
        for t in traverse_unique_terminals(o):
            if isinstance(t, Argument):
                raise ArityMismatch(
                    f"Applying nonlinear operator {o._ufl_class_.__name__} to "
                    f"expression depending on form argument {t}."
                )
        return self._et

    @process.register(Terminal)
    def terminal(self, o, **kwargs):
        """Apply to terminal."""
        return self._et

    @process.register(Argument)
    def argument(self, o, **kwargs):
        """Apply to argument."""
        return ((o, False),)

    @process.register(Sum)
    @DAGTraverser.postorder
    def sum(self, o, a, b, **kwargs):
        """Apply to sum."""
        if a != b:
            raise ArityMismatch(
                f"Adding expressions with non-matching form arguments "
                f"{tuple(map(_afmt, a))} vs {tuple(map(_afmt, b))}."
            )
        return a

    @process.register(Division)
    @DAGTraverser.postorder
    def division(self, o, a, b, **kwargs):
        """Apply to division."""
        if b:
            raise ArityMismatch(f"Cannot divide by form argument {b}.")
        return a

    @process.register(Product)
    @DAGTraverser.postorder
    def product(self, o, a, b, **kwargs):
        """Apply to product."""
        return self._product(o, a, b)

    def _product(self, o, a, b):
        """Apply product arity rules to processed operands."""
        if a and b:
            # Check that we don't have test*test, trial*trial, even
            # for different parts in a block system
            anumbers = set(x[0].number() for x in a)
            for x in b:
                if x[0].number() in anumbers:
                    raise ArityMismatch(
                        "Multiplying expressions with overlapping form argument number "
                        f"{x[0].number()}, argument is {_afmt(x)}."
                    )
            # Combine argument lists
            c = tuple(sorted(set(a + b), key=lambda x: (x[0].number(), x[0].part())))
            # Check that we don't have any arguments shared between a
            # and b
            if len(c) != len(a) + len(b) or len(c) != len({x[0] for x in c}):
                raise ArityMismatch(
                    "Multiplying expressions with overlapping form arguments "
                    f"{tuple(map(_afmt, a))} vs {tuple(map(_afmt, b))}."
                )
            # It's fine for argument parts to overlap
            return c
        elif a:
            return a
        else:
            return b

    # inner, outer and dot all behave as product but for conjugates
    @process.register(Inner)
    @DAGTraverser.postorder
    def inner(self, o, a, b, **kwargs):
        """Apply to inner."""
        return self._product(o, a, self._conjugate(b))

    @process.register(Dot)
    @DAGTraverser.postorder
    def dot(self, o, a, b, **kwargs):
        """Apply to dot."""
        return self._product(o, a, self._conjugate(b))

    @process.register(Outer)
    @DAGTraverser.postorder
    def outer(self, o, a, b, **kwargs):
        """Apply to outer."""
        return self._product(o, self._conjugate(a), b)

    @process.register(PositiveRestricted)
    @process.register(NegativeRestricted)
    @process.register(CellAvg)
    @process.register(FacetAvg)
    @process.register(Grad)
    @process.register(ReferenceGrad)
    @process.register(NablaGrad)
    @process.register(Div)
    @process.register(ReferenceDiv)
    @process.register(NablaDiv)
    @process.register(Curl)
    @process.register(ReferenceCurl)
    @process.register(ReferenceValue)
    @process.register(Interpolate)
    @DAGTraverser.postorder
    def linear_operator(self, o, a, **kwargs):
        """Apply to linear_operator."""
        return a

    # Conj, is a sesquilinear operator
    @process.register(Conj)
    @DAGTraverser.postorder
    def conj(self, o, a, **kwargs):
        """Apply to conj."""
        return self._conjugate(a)

    @staticmethod
    def _conjugate(a):
        """Toggle the conjugation state of an arity tuple."""
        return tuple((a_[0], not a_[1]) for a_ in a)

    # Does it make sense to have a Variable(Argument)? I see no
    # problem.
    @process.register(Variable)
    @DAGTraverser.postorder
    def variable(self, o, f, a, **kwargs):
        """Apply to variable."""
        return f

    # Conditional is linear on each side of the condition
    @process.register(Conditional)
    @DAGTraverser.postorder
    def conditional(self, o, c, a, b, **kwargs):
        """Apply to conditional."""
        if c:
            raise ArityMismatch("Condition cannot depend on form arguments.")
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
            raise ArityMismatch(
                "Conditional subexpressions with non-matching form arguments "
                f"{tuple(map(_afmt, a))} vs {tuple(map(_afmt, b))}."
            )

    @process.register(Indexed)
    @process.register(IndexSum)
    @process.register(ComponentTensor)
    @DAGTraverser.postorder
    def linear_indexed_type(self, o, a, i, **kwargs):
        """Apply to linear_indexed_type."""
        return a

    @process.register(ListTensor)
    @DAGTraverser.postorder
    def list_tensor(self, o, *ops, **kwargs):
        """Apply to list_tensor."""
        args = set(chain(*ops))
        if args:
            # Check that each list tensor component has the same
            # argument numbers (ignoring parts)
            numbers = set(tuple(sorted(set(arg[0].number() for arg in op))) for op in ops)
            if () in numbers:  # Allow e.g. <v[0], 0, v[1]> but not <v[0], u[0]>
                numbers.remove(())
            if len(numbers) > 1:
                raise ArityMismatch(
                    "Listtensor components must depend on the same argument numbers, "
                    f"found {numbers}."
                )

            # Allow different parts with the same number
            return tuple(sorted(args, key=lambda x: (x[0].number(), x[0].part())))
        else:
            # No argument dependencies
            return self._et


def check_integrand_arity(expr, arguments, complex_mode=False):
    """Check the arity of an integrand.

    Arity extraction records each argument as an ``(argument, is_conjugated)`` pair. This enables
    complex-mode checks to validate the required state: test functions are always conjugated, whereas
    trial functions remain unconjugated.
    """
    arguments = tuple(sorted(set(arguments), key=lambda x: (x.number(), x.part())))
    rules = ArityChecker(arguments)
    arg_tuples = rules(expr)
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
