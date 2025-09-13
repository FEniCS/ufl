import gc
import sys

from utils import LagrangeElement

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
    # The minimum value returned by sys.getrefcount.
    # Python 3.14 introduced borrowing references https://docs.python.org/3.14/glossary.html#term-borrowed-reference.
    # This changed the output of
    #
    #   a = object()
    #   sys.getrefcount(a)
    #
    # from 2 to 1. Avoiding the additional temporary reference increase to pass a to getrefcount.
    MIN_REF_COUNT = 1 if sys.version_info >= (3, 14) else 2

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
    domain = AugmentedMesh(LagrangeElement(cell, 1, (2,)), data=mesh_data)
    element = LagrangeElement(cell, 1)
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

    _stripped_form, mapping = strip_terminal_data(form)

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
    domain = AugmentedMesh(LagrangeElement(cell, 1, (2,)), data=mesh_data)
    element = LagrangeElement(cell, 1)
    V = AugmentedFunctionSpace(domain, element, data=fs_data)

    v = TestFunction(V)
    u = TrialFunction(V)
    f = AugmentedCoefficient(V, data=coeff_data)
    k = AugmentedConstant(V, data=const_data)

    form = k * f * inner(grad(v), grad(u)) * dx
    stripped_form, mapping = strip_terminal_data(form)

    assert stripped_form.signature() == form.signature()
    assert replace_terminal_data(stripped_form, mapping) == form
