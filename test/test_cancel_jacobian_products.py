from utils import FiniteElement, LagrangeElement

from ufl import (
    FunctionSpace,
    Mesh,
    TestFunction,
    TrialFunction,
    conj,
    div,
    dx,
    inner,
    triangle,
)
from ufl.algorithms.analysis import extract_type
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.algorithms.apply_derivatives import apply_derivatives
from ufl.algorithms.apply_function_pullbacks import apply_function_pullbacks
from ufl.algorithms.cancel_jacobian_products import (
    IdentityEliminator,
    JacobianCanceller,
    ReciprocalCanceller,
    cancel_jacobian_products,
)
from ufl.algorithms.compute_form_data import compute_form_data
from ufl.algorithms.remove_component_tensors import remove_component_tensors
from ufl.algorithms.renumbering import renumber_indices
from ufl.classes import (
    Identity,
    Indexed,
    Jacobian,
    JacobianDeterminant,
    JacobianInverse,
    ReferenceGrad,
    ReferenceValue,
)
from ufl.constantvalue import as_ufl
from ufl.core.multiindex import MultiIndex, indices
from ufl.indexsum import IndexSum
from ufl.operators import dot
from ufl.pullback import contravariant_piola
from ufl.sobolevspace import HDiv


def test_jacobian_canceller_and_identity_eliminator():
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    J = Jacobian(domain)
    K = JacobianInverse(domain)

    # cancel_jacobian_products assumes component tensors have already
    # been removed, so contractions appear as IndexSum over Indexed
    # terminals rather than as a Dot operator.
    expr = apply_algebra_lowering(dot(J, K))
    assert JacobianCanceller()(expr) == Identity(2)
    # Idempotent: nothing left to cancel the second time round.
    assert IdentityEliminator()(JacobianCanceller()(expr)) == Identity(2)

    # The other contraction order (K . J) is also a valid Kronecker delta,
    # since K is a left inverse of J even for immersed manifolds.
    expr2 = apply_algebra_lowering(dot(K, J))
    assert JacobianCanceller()(expr2) == Identity(2)


def test_reciprocal_canceller():
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    detJ = JacobianDeterminant(domain)

    expr = detJ**2 * (as_ufl(1) / detJ) ** 2
    assert ReciprocalCanceller()(expr) == as_ufl(1)

    # A reciprocal factor that does not fully cancel is simplified to a
    # single net power, not left alone or over-simplified to nothing.
    expr2 = detJ**3 * (as_ufl(1) / detJ) ** 2
    assert ReciprocalCanceller()(expr2) == detJ


def test_cancel_jacobian_products_on_form():
    """Cancelling J-Jinv contractions in a div-div form over a
    contravariant Piola-mapped element must reproduce the classical
    Piola divergence identity div(u) = (1/detJ) * ref_div(u) exactly --
    not just leave a shorter-looking or differently-typed integrand.
    """
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    RT = FiniteElement("Raviart-Thomas", triangle, 1, (2,), contravariant_piola, HDiv)
    V = FunctionSpace(domain, RT)
    u = TrialFunction(V)
    v = TestFunction(V)
    form = inner(div(u), div(v)) * dx

    # cancel_jacobian_products operates on a derivative-expanded,
    # component-tensor-free integrand -- the same preprocessing
    # compute_form_data applies immediately before calling it.
    f = apply_function_pullbacks(form)
    f = apply_algebra_lowering(f)
    f = apply_derivatives(f)
    f = remove_component_tensors(f)
    integrand = f.integrals()[0].integrand()
    cancelled = cancel_jacobian_products(integrand)

    # The Jacobian and its inverse must be fully eliminated: the only
    # surviving geometric quantity is the reciprocal Jacobian
    # determinant introduced by the Piola scaling, which has nothing
    # left to cancel against on its own.
    assert not extract_type(cancelled, (Jacobian, JacobianInverse))
    assert extract_type(cancelled, JacobianDeterminant)

    # Hand-coded expected result -- the classical Piola divergence
    # identity div(w) = (1/detJ) * ref_div(w), derived independently of
    # cancel_jacobian_products itself and applied to both arguments --
    # rather than any property of cancelled that could hold by accident
    # (e.g. merely being shorter, or merely lacking Jacobian terminals).
    def ref_div(w):
        (k,) = indices(1)
        grad_w = Indexed(ReferenceGrad(ReferenceValue(w)), MultiIndex((k, k)))
        return IndexSum(grad_w, MultiIndex((k,)))

    detJ = JacobianDeterminant(domain)
    expected = (as_ufl(1.0) / detJ * ref_div(u)) * conj(as_ufl(1.0) / detJ * ref_div(v))
    # Independently built expressions carry differently-numbered dummy
    # indices, so renumber both before comparing.
    assert renumber_indices(cancelled) == renumber_indices(expected)

    # do_cancel_jacobian_products defaults to False, so existing form
    # compiler callers (e.g. ffcx) that do not pass it are unaffected.
    fd_kwargs = dict(
        do_apply_function_pullbacks=True,
        do_apply_geometry_lowering=True,
        do_apply_default_restrictions=True,
        do_apply_restrictions=True,
    )
    fd_default = compute_form_data(form, **fd_kwargs)
    fd_plain = compute_form_data(form, do_cancel_jacobian_products=False, **fd_kwargs)
    assert fd_default.preprocessed_form.signature() == fd_plain.preprocessed_form.signature()


def test_cancel_jacobian_products_is_a_no_op_on_affine_elements():
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    V = FunctionSpace(domain, LagrangeElement(triangle, 1))
    u = TrialFunction(V)
    v = TestFunction(V)

    # No Jacobian-inverse contractions or reciprocal Jacobian
    # determinants appear here to begin with, so cancellation must
    # reproduce the input form exactly.
    form = inner(u, v) * dx
    assert cancel_jacobian_products(form).signature() == form.signature()
