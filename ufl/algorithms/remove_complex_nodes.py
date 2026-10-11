"""Remove conj, real, and imag nodes from a form."""

from functools import singledispatchmethod

from ufl.algorithms.map_integrands import map_integrands
from ufl.classes import ComplexValue, Conj, Expr, Imag, Real, Terminal
from ufl.corealg.dag_traverser import DAGTraverser


class ComplexNodeRemoval(DAGTraverser):
    """Replaces complex operator nodes with their children."""

    @singledispatchmethod
    def process(self, o: Expr) -> Expr:
        """Process an expression node."""
        return super().process(o)

    @process.register(Expr)
    @DAGTraverser.postorder
    def _(self, o: Expr, *ops: Expr) -> Expr:
        """Reuse an expression if none of its operands changed."""
        if all(a is b for a, b in zip(o.ufl_operands, ops)):
            return o
        return o._ufl_expr_reconstruct_(*ops)

    @process.register(Terminal)
    def _(self, t: Terminal) -> Terminal:
        """Apply to a terminal."""
        if isinstance(t, ComplexValue):
            raise ValueError("Unexpected complex value in real expression.")
        return t

    @process.register(Conj)
    @DAGTraverser.postorder
    def _(self, o: Conj, a: Expr) -> Expr:
        """Apply to conj."""
        return a

    @process.register(Real)
    @DAGTraverser.postorder
    def _(self, o: Real, a: Expr) -> Expr:
        """Apply to real."""
        return a

    @process.register(Imag)
    @DAGTraverser.postorder
    def _(self, o: Imag, a: Expr) -> Expr:
        """Apply to imag."""
        raise ValueError("Unexpected imag in real expression.")


def remove_complex_nodes(expr):
    """Replaces complex operator nodes with their children.

    This is called during compute_form_data if the compiler wishes to
    compile real-valued forms. In essence this strips all trace of
    complex support from the preprocessed form.
    """
    return map_integrands(ComplexNodeRemoval(), expr)
