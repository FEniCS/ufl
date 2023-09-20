import gc
import sys

from ufl import Coefficient, Constant, FunctionSpace, Mesh, TestFunction, TrialFunction, dx, grad, inner, triangle
from ufl.algorithms import replace_terminal_data, strip_terminal_data
from ufl.core.ufl_id import attach_ufl_id
from ufl.core.ufl_type import attach_operators_from_hash_data
from ufl.finiteelement import FiniteElement
from ufl.sobolevspace import H1

MIN_REF_COUNT = 2
"""The minimum value returned by sys.getrefcount."""


@attach_operators_from_hash_data
@attach_ufl_id
class AugmentedMesh(Mesh):
    def __init__(self, *args, data):
        super().__init__(*args)
        self.data = data


@attach_operators_from_hash_data
class AugmentedFunctionSpace(FunctionSpace):
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
    domain = AugmentedMesh(FiniteElement("Lagrange", cell, 1, (cell.geometric_dimension(), ),
                                         (cell.geometric_dimension(), ), "identity", H1), data=mesh_data)
    element = FiniteElement("Lagrange", cell, 1, (), (), "identity", H1)
    V = AugmentedFunctionSpace(domain, element, data=fs_data)

    v = TestFunction(V)
    u = TrialFunction(V)
    f = AugmentedCoefficient(V, data=coeff_data)
    k = AugmentedConstant(V, data=const_data)

    form = k*f*inner(grad(v), grad(u))*dx

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
    domain = AugmentedMesh(FiniteElement("Lagrange", cell, 1, (cell.geometric_dimension(), ),
                                         (cell.geometric_dimension(), ), "identity", H1), data=mesh_data)
    element = FiniteElement("Lagrange", cell, 1, (), (), "identity", H1)
    V = AugmentedFunctionSpace(domain, element, data=fs_data)

    v = TestFunction(V)
    u = TrialFunction(V)
    f = AugmentedCoefficient(V, data=coeff_data)
    k = AugmentedConstant(V, data=const_data)

    form = k*f*inner(grad(v), grad(u))*dx
    stripped_form, mapping = strip_terminal_data(form)

    assert stripped_form.signature() == form.signature()
    assert replace_terminal_data(stripped_form, mapping) == form
