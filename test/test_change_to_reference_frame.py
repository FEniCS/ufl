"""Tests of the change to reference frame algorithm."""

from ufl import Coefficient, triangle
from ufl.classes import Expr, ReferenceValue
from ufl.finiteelement import FiniteElement
from ufl.sobolevspace import H1


def change_to_reference_frame(expr):
    assert isinstance(expr, Expr)
    return ReferenceValue(expr)


def test_change_unmapped_form_arguments_to_reference_frame():
    U = FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)
    V = FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)
    T = FiniteElement("Lagrange", triangle, 1, (2, 2), (2, 2), "identity", H1)

    expr = Coefficient(U)
    assert change_to_reference_frame(expr) == ReferenceValue(expr)
    expr = Coefficient(V)
    assert change_to_reference_frame(expr) == ReferenceValue(expr)
    expr = Coefficient(T)
    assert change_to_reference_frame(expr) == ReferenceValue(expr)


def test_change_hdiv_form_arguments_to_reference_frame():
    V = FiniteElement("Raviart-Thomas", triangle, 1, (2, ), (2, ), "contravariant Piola", HDiv)

    expr = Coefficient(V)
    assert change_to_reference_frame(expr) == ReferenceValue(expr)


def test_change_hcurl_form_arguments_to_reference_frame():
    V = FiniteElement("Raviart-Thomas", triangle, 1, (2, ), (2, ), "contravariant Piola", HDiv)

    expr = Coefficient(V)
    assert change_to_reference_frame(expr) == ReferenceValue(expr)

    '''
    # user input
    grad(f + g)('+')
    # change to reference frame
    -> (K*rgrad(rv(M)*rv(f) + rv(M)*rv(g)))('+')
    # apply derivatives
    -> (K*(rv(M)*rgrad(rv(f)) + rv(M)*rgrad(rv(g))))('+')
    # apply restrictions
    -> K('+')*(rv(M('+'))*rgrad(rv(f('+'))) + rv(M('+'))*rgrad(rv(g('+'))))



    # user input
    grad(f + g)('+')

    # some derivatives applied before processing
    (grad(f) + grad(g))('+')

    # ... replace to get fully defined form arguments here ...

    # expand compounds
    # * context options:
    #   - keep {types} without rewriting to lower level types
    #   - preserve div and curl if applied directly to terminal
    #     (ffc context may set this to off)
    # * output invariants:
    #   - no compound operator types left in expression (simplified language)
    #   - div and curl rewritten in terms of grad (optionally unless applied directly to terminal)
    -> (grad(f) + grad(g))('+')

    # change to reference frame
    # * context options:
    #   - keep {types} without rewriting to lower level types (e.g. JacobianInverse)
    #     (ffc context may initially add all code snippets expressions)
    #   - keep {types} in global frame (e.g. Coefficient)
    #     (ffc context may initially add Coefficient and Argument here to refrain from changing)
    #   - skip integral scaling
    #     (ffc context may turn skipping on to preserve current behaviour)
    # * output invariants:
    #   - ReferenceValue bound directly to terminals where applicable
    #   - grad replaced by mapping expression of rgrad
    #   - div replaced by mapping expression of rdiv
    #   - curl replaced by mapping expression of rcurl
    -> as_tensor(IndexSum(K[i,j]*rgrad(as_tensor(rv(M)[k,l]*rv(f)[l], (l,))
                                     + as_tensor(rv(M)[r,s]*rv(g)[s], (s,)))[j],
                          j),
                 (i,))('+')

    # apply derivatives
    # * context options:
    #   - N/A?
    # * output invariants:
    #   - grad,div,curl, bound directly to terminals
    #   - rgrad,rdiv,rcurl bound directly to referencevalue objects (rgrad(global_f) invalid)
    -> (K*(rv(M)*rgrad(rv(f)) + rv(M)*rgrad(rv(g))))('+')

    # apply restrictions
    # * context options:
    #   - N/A?
    # * output invariants:
    #   - *_restricted bound directly to terminals
    #   - all terminals that must be restricted to make sense are restricted
    -> K('+')*(rv(M('+'))*rgrad(rv(f('+'))) + rv(M('+'))*rgrad(rv(g('+'))))

    # final modified terminal structure:
    t = terminal | restricted(terminal)  # choice of terminal
    r = rval(t) | rgrad(r)               # in reference frame: value or n-gradient
    g = t | grad(g)                      # in global frame: value or n-gradient
    v = r | g                            # value in either frame
    e = v | cell_avg(v) | facet_avg(v) | at_cell_midpoint(v) | at_facet_midpoint(v)
                                         # evaluated at point or averaged over cell entity
    m = e | indexed(e)                   # scalar component of
    '''


'''
New form preprocessing pipeline:

Preferably introduce these changes:
1) Create new FormArgument Expression without element or domain
2) Create new FormArgument Constant without domain
3) Drop replace
--> but just applying replace first is fine

i) group and join integrals by (domain, type, subdomain_id),
ii) process integrands:
    a) apply_coefficient_completion # replace coefficients to ensure proper elements and domains
    b) lower_compound_operators # expand_compounds
    c) change_to_reference_frame # change f->rv(f), m->M*rv(m),
                                          grad(f)->K*rgrad(rv(f)),
                                          grad(grad(f))->K*rgrad(K*rgrad(rv(f))), grad(expr)->K*rgrad(expr)
                                 # if grad(expr)->K*rgrad(expr) should be valid,
                                   then rgrad must be applicable to quite generic expressions
    d) apply_derivatives         # one possibility is to add an apply_mapped_derivatives AD
                                   algorithm which includes mappings
    e) apply_geometry_lowering
    f) apply_restrictions # requiring grad(f)('+') instead of grad(f('+')) would simplify a lot...
iii) extract final metadata about elements and coefficient ordering
'''
