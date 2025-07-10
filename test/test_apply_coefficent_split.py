from utils import LagrangeElement, MixedElement

from ufl import (
    Coefficient,
    FunctionSpace,
    Mesh,
    MeshSequence,
    triangle,
)
from ufl.algorithms.apply_coefficient_split import apply_coefficient_split
from ufl.classes import (
    ComponentTensor,
    Indexed,
    ListTensor,
    PositiveRestricted,
    ReferenceGrad,
    ReferenceValue,
)


def test_apply_coefficient_split(self):
    cell = triangle
    mesh0 = Mesh(LagrangeElement(cell, 1, (2,)))
    mesh1 = Mesh(LagrangeElement(cell, 1, (2,)))
    elem0 = LagrangeElement(cell, 1)
    elem1 = LagrangeElement(cell, 2)
    V0 = FunctionSpace(mesh0, elem0)
    V1 = FunctionSpace(mesh1, elem1)
    f0 = Coefficient(V0)
    f1 = Coefficient(V1)
    mesh = MeshSequence([mesh0, mesh1])
    elem = MixedElement([elem0, elem1])
    V = FunctionSpace(mesh, elem)
    f = Coefficient(V)
    coefficient_split = {f: (f0, f1)}
    expr = PositiveRestricted(ReferenceGrad(ReferenceValue(f)))
    expr_split = apply_coefficient_split(expr, coefficient_split)
    # Below is ``repr(expr_split)`` to be checked:
    #
    # expr_split = ComponentTensor(
    #     Indexed(
    #         ListTensor(
    #             Indexed(
    #                 PositiveRestricted(ReferenceGrad(ReferenceValue(f0))), MultiIndex((idx1,))
    #             ),
    #             Indexed(
    #                 PositiveRestricted(ReferenceGrad(ReferenceValue(f1))), MultiIndex((idx1,))
    #             ),
    #         ),
    #         MultiIndex((idx0,))
    #     ),
    #     MultiIndex((idx0, idx1))
    # )
    assert isinstance(expr_split, ComponentTensor)
    op, (idx0, idx1) = expr_split.ufl_operands
    assert isinstance(op, Indexed)
    op_, (idx0_,) = op.ufl_operands
    assert isinstance(op_, ListTensor)
    assert idx0_ == idx0
    op0, op1 = op_.ufl_operands
    op0_, (idx1_,) = op0.ufl_operands
    assert op0_ == PositiveRestricted(ReferenceGrad(ReferenceValue(f0)))
    assert idx1_ == idx1
    op1_, (idx1_,) = op1.ufl_operands
    assert op1_ == PositiveRestricted(ReferenceGrad(ReferenceValue(f1)))
    assert idx1_ == idx1
