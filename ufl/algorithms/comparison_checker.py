"""Algorithm to check for 'comparison' nodes in a form when the user is in 'complex mode'."""

from functools import singledispatchmethod

from ufl.algebra import Abs, Imag, Power, Real
from ufl.algorithms.map_integrands import map_integrands
from ufl.argument import Argument
from ufl.conditional import GE, GT, LE, LT, MaxValue, MinValue
from ufl.constantvalue import RealValue, Zero
from ufl.core.expr import Expr
from ufl.core.multiindex import MultiIndex
from ufl.core.terminal import Terminal
from ufl.corealg.dag_traverser import DAGTraverser
from ufl.geometry import GeometricQuantity
from ufl.indexed import Indexed
from ufl.mathfunctions import Sqrt


class CheckComparisons(DAGTraverser):
    """Raises an error if comparisons are done with complex quantities.

    If quantities are real, adds the Real operator to the compared quantities.

    Terminals that are real are RealValue, Zero, and Argument
    (even in complex FEM, the basis functions are real)
    Operations that produce reals are Abs, Real, Imag.
    Scalar terminals default to complex, and Sqrt, Pow (defensively) imply complex.
    Indexing metadata does not affect scalar type propagation.
    Otherwise, operators preserve the type of their operands.
    """

    def __init__(self, **kwargs):
        """Initialise."""
        super().__init__(**kwargs)
        self.nodetype = {}

    @singledispatchmethod
    def process(self, o: Expr, **kwargs):
        """Process an expression node."""
        return super().process(o, **kwargs)

    def _reuse_if_untouched(self, o, *operands):
        """Reuse ``o`` if its processed operands are unchanged."""
        if all(a is b for a, b in zip(o.ufl_operands, operands)):
            return o
        return o._ufl_expr_reconstruct_(*operands)

    @process.register(Expr)
    @DAGTraverser.postorder
    def _(self, o, *operands, **kwargs):
        """Defaults expressions to complex unless they only act on real quantities.

        Overridden for specific operators. Rebuilds objects if necessary.
        """
        types = {self.nodetype[operand] for operand in operands}

        if types:
            t = "complex" if "complex" in types else "real"
        else:
            t = "complex"

        result = self._reuse_if_untouched(o, *operands)
        self.nodetype[result] = t
        return result

    @process.register(LT)
    @process.register(GT)
    @process.register(LE)
    @process.register(GE)
    @DAGTraverser.postorder
    def _(self, o, *operands, **kwargs):
        """Compare."""
        types = {self.nodetype[operand] for operand in operands}

        if "complex" in types:
            raise ComplexComparisonError("Ordering undefined for complex values.")
        else:
            result = o._ufl_expr_reconstruct_(*map(Real, operands))
            self.nodetype[result] = "bool"
            return result

    @process.register(MaxValue)
    @process.register(MinValue)
    @DAGTraverser.postorder
    def _(self, o, *operands, **kwargs):
        """Apply to max_value."""
        types = {self.nodetype[operand] for operand in operands}

        if "complex" in types:
            name = "max" if isinstance(o, MaxValue) else "min"
            raise ComplexComparisonError(f"You can't compare complex numbers with {name}.")
        else:
            result = o._ufl_expr_reconstruct_(*map(Real, operands))
            self.nodetype[result] = "bool"
            return result

    @process.register(Real)
    @process.register(Imag)
    @DAGTraverser.postorder
    def _(self, o, *operands, **kwargs):
        """Apply to real."""
        result = self._reuse_if_untouched(o, *operands)
        self.nodetype[result] = "real"
        return result

    @process.register(Sqrt)
    @DAGTraverser.postorder
    def _(self, o, *operands, **kwargs):
        """Apply to sqrt."""
        result = self._reuse_if_untouched(o, *operands)
        self.nodetype[result] = "complex"
        return result

    @process.register(Power)
    @DAGTraverser.postorder
    def _(self, o, base, exponent, **kwargs):
        """Apply to power."""
        result = self._reuse_if_untouched(o, base, exponent)
        try:
            # Attempt to diagnose circumstances in which the result must be real.
            exponent = float(exponent)
            if self.nodetype[base] == "real" and int(exponent) == exponent:
                self.nodetype[result] = "real"
                return result
        except TypeError:
            pass

        self.nodetype[result] = "complex"
        return result

    @process.register(Abs)
    @DAGTraverser.postorder
    def _(self, o, *operands, **kwargs):
        """Apply to abs."""
        result = self._reuse_if_untouched(o, *operands)
        self.nodetype[result] = "real"
        return result

    @process.register(Terminal)
    def _(self, term, **kwargs):
        """Apply to terminal."""
        # default terminals to complex, except the ones we *know* are real
        if isinstance(term, RealValue | Zero | Argument | GeometricQuantity):
            self.nodetype[term] = "real"
        else:
            self.nodetype[term] = "complex"
        return term

    @process.register(MultiIndex)
    def _(self, index, **kwargs):
        """Ignore indexing metadata when determining scalar type."""
        self.nodetype[index] = "real"
        return index

    @process.register(Indexed)
    @DAGTraverser.postorder
    def _(self, o, expr, multiindex, **kwargs):
        """Apply to indexed."""
        result = self._reuse_if_untouched(o, expr, multiindex)
        self.nodetype[result] = self.nodetype[expr]
        return result


def do_comparison_check(form):
    """Raises an error if invalid comparison nodes exist."""
    return map_integrands(CheckComparisons(), form)


class ComplexComparisonError(BaseException):
    """Complex compariseon exception."""
