"""Tools to extract linear combinations from UFL expressions."""

# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from functools import singledispatchmethod

import ufl
from ufl.algorithms.remove_component_tensors import remove_component_tensors
from ufl.corealg.dag_traverser import DAGTraverser


class LinearCombinationExtractor(DAGTraverser):
    """Bottom-up DAG traverser for extracting linear combinations.

    To process an arbitrary mathematical expression, this traverser categorizes
    every node in the DAG into one of two states, returning different types for each:

    1. Scalar Weights (Returns: {py:class}`ufl.core.expr.Expr`)
       If a node and all its children represent a global scalar value (e.g.,
       {py:class}`ufl.FloatValue`, {py:class}`ufl.Constant`), the traverser
       propagates the actual UFL expression upwards. It does not evaluate them
       to Python floats, preserving the full UFL AST of the constants.

    2. Spatial Fields (Returns: `list[tuple[ufl.core.expr.Expr, ufl.Coefficient]]`)
       If a node contains spatial functions (standard Coefficients), it must
       maintain the strict algebraic structure of a linear combination. Therefore,
       it returns a list of `(weight, function)` tuples, where `weight` is the
       accumulated UFL expression and `function` is the base spatial field.

    By strictly distinguishing between the two (checking `isinstance(..., list)`),
    the traverser can safely apply algebraic rules (e.g., multiplying a list by a
    scalar weight expression distributes the weight) and instantly catch illegal
    non-linear operations (e.g., attempting to multiply two lists together).
    """

    def __init__(self, **kwargs):
        """Initialize LinearCombinationExtractor with memoization and no compression.

        Compression is disabled to avoid hashing unhashable return types (like lists)
        while preserving the `_visited_cache` memoization.
        """
        kwargs["compress"] = False
        super().__init__(**kwargs)

    @singledispatchmethod
    def process(self, o: ufl.classes.Expr, **kwargs):
        """Fallback for any unsupported node types."""
        raise ValueError(f"Unsupported UFL node type for linear combinations: {type(o)}")

    # ---------------------------------------------------------
    # 1. Terminals (Leaves) - No children to evaluate
    # ---------------------------------------------------------
    @process.register(ufl.classes.IntValue)
    @process.register(ufl.classes.FloatValue)
    @process.register(ufl.classes.ScalarValue)
    def _(self, o, **kwargs):
        # Return the UFL expression itself
        return o

    @process.register(ufl.classes.Zero)
    def _(self, o, **kwargs):
        return o if o.ufl_shape == () else []

    @process.register(ufl.Constant)
    def _(self, o, **kwargs):
        if o.ufl_shape == ():
            return o
        raise ValueError(f"Only scalar constants are supported, got shape {o.ufl_shape}")

    @process.register(ufl.Matrix)
    @process.register(ufl.coefficient.BaseCoefficient)
    def _(self, o, **kwargs):
        # Check for real-valued elements
        if isinstance(o, ufl.core.expr.Expr) and ufl.checks.is_scalar_constant_expression(o):
            return o
        return [(ufl.as_ufl(1.0), o)]

    # ---------------------------------------------------------
    # 2. Operators - Use @postorder to evaluate operands first
    # ---------------------------------------------------------
    @process.register(ufl.classes.Sum)
    @DAGTraverser.postorder
    def _(self, o, *operands, **kwargs):
        # If no operands are lists, this is a pure scalar addition.
        # We construct a new UFL expression by safely summing them.
        if all(not isinstance(op, list) for op in operands):
            res = operands[0]
            for op in operands[1:]:
                res = res + op
            return res

        # Otherwise, accumulate the spatial functions
        res = []
        for op_res in operands:
            if isinstance(op_res, list):
                res.extend(op_res)
            else:
                raise ValueError(
                    "Cannot directly add a raw scalar expression to a spatial function."
                )
        return res

    @process.register(ufl.Action)
    def _(self, o, **kwargs):
        # An Action node represents a matrix-vector product (e.g., A * u).
        # This cannot be reduced to a simple algebraic linear combination of arrays.
        raise ValueError("Non-linear expression detected: product of two spatial functions.")

    @process.register(ufl.classes.FormSum)
    @process.register(ufl.form.FormSum)
    def _(self, o, **kwargs):
        res = []
        components = o.components()
        weights = o.weights()
        for weight, comp in zip(weights, components):
            # Evaluate the base component (e.g., Matrix or Cofunction)
            comp_res = self(comp, **kwargs)

            # Evaluate the weight (in case it contains sub-expressions)
            w_res = self(weight, **kwargs) if isinstance(weight, ufl.classes.Expr) else weight

            if isinstance(comp_res, list):
                # Distribute this FormSum weight into the component's linear combination
                res.extend([(w_res * w, f) for w, f in comp_res])
            else:
                raise ValueError(
                    "Cannot directly add a raw scalar expression to a spatial function."
                )

        return res

    @process.register(ufl.classes.Product)
    @DAGTraverser.postorder
    def _(self, o, *operands, **kwargs):
        op1_res, op2_res = operands
        # Each of the operands are either a scalar UFL expression (float, Constant, etc.)
        # or a list of (weight, function) tuples.
        # The following cases are possible:
        # 1. Both operands are scalars: return the product of the two UFL expressions.
        # 2. One operand is a scalar, the other is a list: distribute the scalar across the list.
        # 3. Both operands are lists: this is a non-linear operation and should raise an error.
        is_list1 = isinstance(op1_res, list)
        is_list2 = isinstance(op2_res, list)
        if not is_list1 and not is_list2:
            return op1_res * op2_res  # UFL operator overloading takes over
        elif not is_list1 and is_list2:
            return [(op1_res * w, f) for w, f in op2_res]
        elif not is_list2 and is_list1:
            return [(op2_res * w, f) for w, f in op1_res]
        else:
            raise ValueError("Non-linear expression detected: product of two spatial functions.")

    @process.register(ufl.classes.Division)
    @DAGTraverser.postorder
    def _(self, o, *operands, **kwargs):
        num_res, den_res = operands
        if isinstance(den_res, list):
            raise ValueError("Non-linear expression detected: division by a spatial function.")

        if not isinstance(num_res, list):
            return num_res / den_res
        return [(w / den_res, f) for w, f in num_res]

    @process.register(ufl.classes.Power)
    @DAGTraverser.postorder
    def _(self, o, *operands, **kwargs):
        base_res, exp_res = operands
        if isinstance(base_res, list) or isinstance(exp_res, list):
            raise ValueError("Non-linear expression detected: power involving a spatial function.")
        return base_res**exp_res

    # ---------------------------------------------------------
    # 3. Tensor Components (Vector-Scalar multiplication)
    # ---------------------------------------------------------
    @process.register(ufl.classes.MultiIndex)
    def _(self, o, **kwargs):
        return o

    @process.register(ufl.classes.Indexed)
    @DAGTraverser.postorder
    def _(self, o, *operands, **kwargs):
        base_res, indices_res = operands

        # We ONLY allow UFL's free indices. Fixed components (FixedIndex)
        # or slices (SliceIndex) do not represent the whole operator.
        if not all(isinstance(idx, ufl.classes.Index) for idx in indices_res.indices()):
            raise ValueError(
                "Explicit component indexing (e.g., u[0]) is not supported. "
                "Only full-operator free indices are allowed."
            )

        if isinstance(base_res, list):
            # Temporarily return the Indexed spatial field.
            # The ComponentTensor node higher up the tree will collapse this
            # back into the un-indexed base operator using remove_component_tensors.
            return [(w, ufl.classes.Indexed(f, indices_res)) for w, f in base_res]

        return ufl.classes.Indexed(base_res, indices_res)

    @process.register(ufl.classes.ComponentTensor)
    @DAGTraverser.postorder
    def _(self, o, *operands, **kwargs):
        expr_res, indices_res = operands

        if isinstance(expr_res, list):
            res = []
            for w, f in expr_res:
                # f is something like Indexed(stage_0, i). Reconstruct the tensor:
                reconstructed_func = ufl.classes.ComponentTensor(f, indices_res)

                # Collapse it back to the base operator (e.g., stage_0)
                simplified_func = remove_component_tensors(reconstructed_func)

                # Enforce that the result actually resolved back to a whole operator
                if isinstance(simplified_func, (ufl.classes.Indexed, ufl.classes.ComponentTensor)):
                    raise ValueError(
                        "Indexing did not cleanly resolve to a complete spatial operator."
                    )

                res.append((w, simplified_func))
            return res

        return ufl.classes.ComponentTensor(expr_res, indices_res)


def extract_linear_combination(
    expr: ufl.core.expr.Expr | ufl.form.BaseForm,
) -> list[tuple[ufl.core.expr.Expr, ufl.coefficient.BaseCoefficient]]:
    """Wrapper to initialize traverser and extract linear combinations.

    Returns:
        A list of tuples where the first element is the UFL expression of the
        weight, and the second element is the base UFL Coefficient (spatial function).
    """
    extractor = LinearCombinationExtractor()
    final_result = extractor(expr)

    if not isinstance(final_result, list):
        raise ValueError("Expression evaluated to a pure scalar, no spatial functions found.")

    return final_result
