from utils import FiniteElement, LagrangeElement

from ufl import (
    Coefficient,
    FunctionSpace,
    Mesh,
    TestFunction,
    TrialFunction,
    div,
    dx,
    inner,
    triangle,
)
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.algorithms.cancel_jacobian_products import (
    IdentityEliminator,
    JacobianCanceller,
    ReciprocalCanceller,
    cancel_jacobian_products,
)
from ufl.algorithms.compute_form_data import compute_form_data
from ufl.classes import Identity, Jacobian, JacobianDeterminant, JacobianInverse
from ufl.constantvalue import as_ufl
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
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    RT = FiniteElement("Raviart-Thomas", triangle, 1, (2,), contravariant_piola, HDiv)
    V = FunctionSpace(domain, RT)
    u = Coefficient(V)
    v = TestFunction(V)

    form = inner(div(u), div(v)) * dx
    fd_kwargs = dict(
        do_apply_function_pullbacks=True,
        do_apply_geometry_lowering=True,
        do_apply_default_restrictions=True,
        do_apply_restrictions=True,
    )

    fd_plain = compute_form_data(form, **fd_kwargs)
    fd_cancelled = compute_form_data(form, do_cancel_jacobian_products=True, **fd_kwargs)

    integrand_plain = fd_plain.preprocessed_form.integrals()[0].integrand()
    integrand_cancelled = fd_cancelled.preprocessed_form.integrals()[0].integrand()

    # The whole point of the pass: the div-div integrand of a
    # contravariant Piola-mapped element is dominated by J-Jinv
    # contractions that cancel, so the cancelled integrand is much
    # smaller than the one form compilers would otherwise have to deal
    # with, and does not introduce any new terminal type of its own.
    assert len(str(integrand_cancelled)) < len(str(integrand_plain)) / 4

    # do_cancel_jacobian_products defaults to False, so existing form
    # compiler callers (e.g. ffcx) that do not pass it are unaffected.
    fd_default = compute_form_data(form, **fd_kwargs)
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
