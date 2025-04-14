from ufl import (
    Coefficient,
    FunctionSpace,
    Measure,
    Mesh,
    MixedFunctionSpace,
    TestFunctions,
    TrialFunctions,
    interval,
    tetrahedron,
    triangle,
)
from ufl.classes import ComponentTensor, Index, Indexed, ListTensor, MultiIndex, NegativeRestricted, PositiveRestricted, ReferenceGrad, ReferenceValue
from ufl.algorithms.formsplitter import extract_blocks
from ufl.algorithms.apply_coefficient_split import apply_coefficient_split
from ufl.finiteelement import FiniteElement
from ufl.pullback import identity_pullback
from ufl.sobolevspace import H1
from ufl.finiteelement import FiniteElement, MixedElement


def test_apply_coefficient_split(self):
    cell = triangle
    mesh = Mesh(FiniteElement("Lagrange", cell, 1, (2,), identity_pullback, H1))
    elem0 = FiniteElement("Lagrange", cell, 1, (), identity_pullback, H1)
    elem1 = FiniteElement("Lagrange", cell, 2, (), identity_pullback, H1)
    elem = MixedElement([elem0, elem1])
    V0 = FunctionSpace(mesh, elem0)
    V1 = FunctionSpace(mesh, elem1)
    V = FunctionSpace(mesh, elem)
    f0 = Coefficient(V0)
    f1 = Coefficient(V1)
    f = Coefficient(V)
    coefficient_split = {f: (f0, f1)}
    expr = PositiveRestricted(ReferenceGrad(ReferenceValue(f)))
    expr_split = apply_coefficient_split(expr, coefficient_split)
    # expr_split = ComponentTensor(
    #     Indexed(
    #         ListTensor(
    #             Indexed(PositiveRestricted(ReferenceGrad(ReferenceValue(f0))), MultiIndex((idx1,))),
    #             Indexed(PositiveRestricted(ReferenceGrad(ReferenceValue(f1))), MultiIndex((idx1,)))
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
