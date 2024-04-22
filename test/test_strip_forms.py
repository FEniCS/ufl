import gc
import sys

from ufl import (
    Coefficient,
    Constant,
    FunctionSpace,
    Mesh,
    TestFunction,
    TrialFunction,
    dx,
    grad,
    inner,
    triangle,
)
from ufl.algorithms import replace_terminal_data, strip_terminal_data
from ufl.core.ufl_id import attach_ufl_id
from ufl.core.ufl_type import UFLObject
from ufl.finiteelement import FiniteElement
from ufl.pullback import identity_pullback
from ufl.sobolevspace import H1

MIN_REF_COUNT = 2
"""The minimum value returned by sys.getrefcount."""


@attach_ufl_id
class AugmentedMesh(Mesh, UFLObject):
    def __init__(self, *args, data):
        super().__init__(*args)
        self.data = data


class AugmentedFunctionSpace(FunctionSpace, UFLObject):
    def __init__(self, *args, data):
        super().__init__(*args)
        self.data = data


class AugmentedCoefficient(Coefficient):
    def __init__(self, *args, data):
        super().__init__(*args)
        self.data = data


class AugmentedConstant(Constant):
    def __init__(self, *args, data):
        super().__init__(*args)
        self.data = data


def test_strip_form_arguments_strips_data_refs():
    mesh_data = object()
    fs_data = object()
    coeff_data = object()
    const_data = object()

    # Sanity check
    assert sys.getrefcount(mesh_data) == MIN_REF_COUNT
    assert sys.getrefcount(fs_data) == MIN_REF_COUNT
    assert sys.getrefcount(coeff_data) == MIN_REF_COUNT
    assert sys.getrefcount(const_data) == MIN_REF_COUNT

    cell = triangle
    domain = AugmentedMesh(
        FiniteElement("Lagrange", cell, 1, (2,), identity_pullback, H1), data=mesh_data
    )
    element = FiniteElement("Lagrange", cell, 1, (), identity_pullback, H1)
    V = AugmentedFunctionSpace(domain, element, data=fs_data)

    v = TestFunction(V)
    u = TrialFunction(V)
    f = AugmentedCoefficient(V, data=coeff_data)
    k = AugmentedConstant(V, data=const_data)

    form = k * f * inner(grad(v), grad(u)) * dx

    # Remove extraneous references
    del cell, domain, element, V, v, u, f, k

    assert sys.getrefcount(mesh_data) == MIN_REF_COUNT + 1
    assert sys.getrefcount(fs_data) == MIN_REF_COUNT + 1
    assert sys.getrefcount(coeff_data) == MIN_REF_COUNT + 1
    assert sys.getrefcount(const_data) == MIN_REF_COUNT + 1

    stripped_form, mapping = strip_terminal_data(form)

    del form, mapping
    gc.collect()  # This is needed to update the refcounts

    assert sys.getrefcount(mesh_data) == MIN_REF_COUNT
    assert sys.getrefcount(fs_data) == MIN_REF_COUNT
    assert sys.getrefcount(coeff_data) == MIN_REF_COUNT
    assert sys.getrefcount(const_data) == MIN_REF_COUNT


def test_strip_form_arguments_does_not_change_form():
    mesh_data = object()
    fs_data = object()
    coeff_data = object()
    const_data = object()

    cell = triangle
    domain = AugmentedMesh(
        FiniteElement("Lagrange", cell, 1, (2,), identity_pullback, H1), data=mesh_data
    )
    element = FiniteElement("Lagrange", cell, 1, (), identity_pullback, H1)
    V = AugmentedFunctionSpace(domain, element, data=fs_data)

    v = TestFunction(V)
    u = TrialFunction(V)
    f = AugmentedCoefficient(V, data=coeff_data)
    k = AugmentedConstant(V, data=const_data)

    form = k * f * inner(grad(v), grad(u)) * dx
    stripped_form, mapping = strip_terminal_data(form)

    assert stripped_form.signature() == form.signature()
    assert replace_terminal_data(stripped_form, mapping) == form
