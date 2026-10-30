"""Algorithm for expanding compound expressions."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs and Anders Logg
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2009-2010

from functools import singledispatchmethod

import ufl.classes
from ufl.algorithms.map_integrands import map_integrands
from ufl.classes import Conj, Grad, Product
from ufl.compound_expressions import cofactor_expr, determinant_expr, deviatoric_expr, inverse_expr
from ufl.core.base_form_operator import BaseFormOperator
from ufl.core.multiindex import Index, indices
from ufl.corealg.dag_traverser import DAGTraverser
from ufl.differentiation import BaseFormOperatorDerivative
from ufl.form import Form
from ufl.tensors import as_matrix, as_tensor, as_vector


class LowerCompoundAlgebra(DAGTraverser):
    """Expands high level compound operators to equivalent representations using basic operators."""

    @singledispatchmethod
    def process(self, o: ufl.classes.Expr) -> ufl.classes.Expr:
        """Process ``o``."""
        return super().process(o)

    @process.register(ufl.classes.Expr)
    @process.register(ufl.classes.BaseForm)
    def _(self, o: ufl.classes.Expr) -> ufl.classes.Expr:
        return self.reuse_if_untouched(o)

    @process.register(Form)
    def _(self, o: Form) -> Form:
        """Lower tensor algebra in the integrands of a nested Form."""
        return map_integrands(self, o)

    @process.register(BaseFormOperator)
    def _(self, o: BaseFormOperator) -> BaseFormOperator:
        """Lower tensor algebra in an operator's operands and argument slots."""
        operands = tuple(self(op) for op in o.ufl_operands)
        slots = tuple(self(slot) for slot in o.argument_slots())
        if operands == o.ufl_operands and slots == o.argument_slots():
            return o
        return o._ufl_expr_reconstruct_(*operands, argument_slots=slots)

    @process.register(BaseFormOperatorDerivative)
    def _(self, o: BaseFormOperatorDerivative) -> BaseFormOperatorDerivative:
        """Traverse the operands that define an operator derivative."""
        return self.reuse_if_untouched(o)

    # ------------ Compound tensor operators

    @process.register(ufl.classes.Trace)
    @DAGTraverser.postorder
    def _(self, o, A):
        """Lower a trace."""
        i = Index()
        return A[i, i]

    @process.register(ufl.classes.Transposed)
    @DAGTraverser.postorder
    def _(self, o, A):
        """Lower a transposed."""
        i, j = indices(2)
        return as_tensor(A[i, j], (j, i))

    @process.register(ufl.classes.Deviatoric)
    @DAGTraverser.postorder
    def _(self, o, A):
        """Lower a deviatoric."""
        return deviatoric_expr(A)

    @process.register(ufl.classes.Skew)
    @DAGTraverser.postorder
    def _(self, o, A):
        """Lower a skew."""
        i, j = indices(2)
        return as_matrix((A[i, j] - A[j, i]) / 2, (i, j))

    @process.register(ufl.classes.Sym)
    @DAGTraverser.postorder
    def _(self, o, A):
        """Lower a sym."""
        i, j = indices(2)
        return as_matrix((A[i, j] + A[j, i]) / 2, (i, j))

    @process.register(ufl.classes.Cross)
    @DAGTraverser.postorder
    def _(self, o, a, b):
        """Lower a cross."""

        def c(i, j):
            return Product(a[i], b[j]) - Product(a[j], b[i])

        return as_vector((c(1, 2), c(2, 0), c(0, 1)))

    @process.register(ufl.classes.Perp)
    @DAGTraverser.postorder
    def _(self, o, a):
        """Lower a perp."""
        return as_vector([-a[1], a[0]])

    @process.register(ufl.classes.Dot)
    @DAGTraverser.postorder
    def _(self, o, a, b):
        """Lower a dot."""
        ai = indices(len(a.ufl_shape) - 1)
        bi = indices(len(b.ufl_shape) - 1)
        k = (Index(),)
        # Creates a single IndexSum over a Product
        s = a[ai + k] * b[k + bi]
        return as_tensor(s, ai + bi)

    @process.register(ufl.classes.Inner)
    @DAGTraverser.postorder
    def _(self, o, a, b):
        """Lower an inner."""
        ash = a.ufl_shape
        bsh = b.ufl_shape
        if ash != bsh:
            raise ValueError("Nonmatching shapes.")
        ii = indices(len(ash))
        # Creates multiple IndexSums over a Product
        s = a[ii] * Conj(b[ii])
        return s

    @process.register(ufl.classes.Outer)
    @DAGTraverser.postorder
    def _(self, o, a, b):
        """Lower an outer."""
        ii = indices(len(a.ufl_shape))
        jj = indices(len(b.ufl_shape))
        # Create a Product with no shared indices
        s = Conj(a[ii]) * b[jj]
        return as_tensor(s, ii + jj)

    @process.register(ufl.classes.Determinant)
    @DAGTraverser.postorder
    def _(self, o, A):
        """Lower a determinant."""
        return determinant_expr(A)

    @process.register(ufl.classes.Cofactor)
    @DAGTraverser.postorder
    def _(self, o, A):
        """Lower a cofactor."""
        return cofactor_expr(A)

    @process.register(ufl.classes.Inverse)
    @DAGTraverser.postorder
    def _(self, o, A):
        """Lower an inverse."""
        return inverse_expr(A)

    # ------------ Compound differential operators

    @process.register(ufl.classes.Div)
    @DAGTraverser.postorder
    def _(self, o, a):
        """Lower a div."""
        i = Index()
        return a[..., i].dx(i)

    @process.register(ufl.classes.NablaDiv)
    @DAGTraverser.postorder
    def _(self, o, a):
        """Lower a nabla_div."""
        i = Index()
        return a[i, ...].dx(i)

    @process.register(ufl.classes.NablaGrad)
    @DAGTraverser.postorder
    def _(self, o, a):
        """Lower a nabla_grad."""
        sh = a.ufl_shape
        if sh == ():
            return Grad(a)
        else:
            j = Index()
            ii = tuple(indices(len(sh)))
            return as_tensor(a[ii].dx(j), (j,) + ii)

    @process.register(ufl.classes.Curl)
    @DAGTraverser.postorder
    def _(self, o, a):
        """Lower a curl."""

        # o = curl a = "[a.dx(1), -a.dx(0)]"            if a.ufl_shape == ()
        # o = curl a = "cross(nabla, (a0, a1, 0))[2]" if a.ufl_shape == (2,)
        # o = curl a = "cross(nabla, a)"              if a.ufl_shape == (3,)
        def c(i, j):
            """A component of curl."""
            return a[j].dx(i) - a[i].dx(j)

        sh = a.ufl_shape
        if sh == ():
            return as_vector((a.dx(1), -a.dx(0)))
        if sh == (2,):
            return c(0, 1)
        if sh == (3,):
            return as_vector((c(1, 2), c(2, 0), c(0, 1)))
        raise ValueError(f"Invalid shape {sh} of curl argument.")


def apply_algebra_lowering(
    expr: ufl.classes.Expr | ufl.classes.BaseForm, preserve_types: tuple[type, ...] = ()
) -> ufl.classes.Expr | ufl.classes.BaseForm:
    """Expand tensor algebra into index notation.

    Args:
        expr: The expression whose compound operators are to be lowered.
        preserve_types: Types whose operators are preserved while their operands are lowered.
            Subclasses of these types are also preserved.

    Returns:
        The expression with compound tensor operators lowered.
    """
    return map_integrands(LowerCompoundAlgebra(preserve_types=preserve_types), expr)
