#!/usr/bin/env py.test

from pytest import raises
from ufl import *
from ufl.algorithms.apply_function_pullbacks import apply_function_pullbacks
from ufl.algorithms import renumber_indices
from ufl.classes import Jacobian, JacobianInverse, JacobianDeterminant, ReferenceValue

def check_form_pullback(form, mappings):
    if not renumber_indices(apply_function_pullbacks(form)) == renumber_indices(replace(form, mappings)):
        print
        print "In check_form_pullback:"
        print
        print str(renumber_indices(form))
        print
        print str(renumber_indices(replace(form, mappings)))
        print
        print str(renumber_indices(apply_function_pullbacks(form)))
        print
        print replace(form, mappings).signature()
        print apply_function_pullbacks(form).signature()
        print
    assert apply_function_pullbacks(form).signature() == replace(form, mappings).signature()
    assert renumber_indices(apply_function_pullbacks(form)) == renumber_indices(replace(form, mappings))

def test_apply_function_pullbacks_keeps_scalar_coefficients_simple():
    cell = triangle

    V0 = FiniteElement("DG", cell, 0)
    V1 = FiniteElement("Lagrange", cell, 1)
    V2 = FiniteElement("Lagrange", cell, 2)

    f0 = Coefficient(V0)
    f = Coefficient(V1)
    g = Coefficient(V2)

    f0r = ReferenceValue(f0)
    fr = ReferenceValue(f)
    gr = ReferenceValue(g)

    # Test that simple forms with scalar coefficients gain no complexity
    mappings = {
        f0: f0r,
        f: fr,
        g: gr,
        }

    check_form_pullback(f0*dx, mappings)
    check_form_pullback(f*dx, mappings)
    check_form_pullback(g*ds, mappings)
    check_form_pullback((f**f0 + g/f)*dx, mappings)

def test_apply_function_pullbacks_keeps_vector_lagrange_coefficients_simple():
    cell = triangle

    V0 = VectorElement("DG", cell, 0)
    V1 = VectorElement("Lagrange", cell, 1)
    V2 = VectorElement("Lagrange", cell, 2)

    f0 = Coefficient(V0)
    f = Coefficient(V1)
    g = Coefficient(V2)

    f0r = ReferenceValue(f0)
    fr = ReferenceValue(f)
    gr = ReferenceValue(g)

    # Test that simple forms with vector coefficients gain no complexity
    mappings = {
        f0: f0r,
        f: fr,
        g: gr,
        }

    check_form_pullback(f0**2*dx, mappings)
    check_form_pullback(f**2*dx, mappings)
    check_form_pullback(g[0]*dx, mappings)
    check_form_pullback((f0[0]*f**2 + g**2/f**2)*dx, mappings)

def test_apply_function_pullbacks_keeps_tensor_lagrange_coefficients_simple():
    cell = triangle

    V0 = TensorElement("DG", cell, 0)
    V1 = TensorElement("Lagrange", cell, 1)
    V2 = TensorElement("Lagrange", cell, 2)

    f0 = Coefficient(V0)
    f = Coefficient(V1)
    g = Coefficient(V2)

    f0r = ReferenceValue(f0)
    fr = ReferenceValue(f)
    gr = ReferenceValue(g)

    # Test that simple forms with vector coefficients gain no complexity
    mappings = {
        f0: f0r,
        f: fr,
        g: gr,
        }

    check_form_pullback(f0**2*dx, mappings)
    check_form_pullback(f**2*dx, mappings)
    check_form_pullback(g[0,0]*dx, mappings)
    check_form_pullback((f0[0,0]*f**2 + g**2/f**2)*dx, mappings)

def test_apply_function_pullbacks_on_mixed_lagrange_functions_stays_simple():
    cell = triangle

    V0 = FiniteElement("DG", cell, 0)
    V1 = VectorElement("Lagrange", cell, 1)
    V2 = TensorElement("Lagrange", cell, 2)
    V3 = TensorElement("Lagrange", cell, 3, symmetry=True)
    W = (V0*V1)*(V2*V3)

    f = Coefficient(W)
    f01, f23 = split(f)
    fr = ReferenceValue(f)

    assert product(f.ufl_shape) == product(f01.ufl_shape) + product(f23.ufl_shape)
    assert product(f.ufl_shape) == 1+2+4+4
    #assert product(fr.ufl_shape) == 1+2+4+3 # FIXME: ReferenceValue doesn't heed symmetry

    # Test that simple forms with mixed Lagrange coefficients gain no complexity
    mappings = {
        f: fr,
        }

    check_form_pullback(f**2*dx, mappings)
    check_form_pullback(f[0]**2*dx, mappings)
    check_form_pullback(f01**2*dx, mappings)
    check_form_pullback(f23**2*dx, mappings)

def test_apply_function_pullbacks_applies_piola_mappings_to_single_mapped_functions():
    cell = triangle
    domain = as_domain(cell)

    V1 = FiniteElement("RT", cell, 1)
    V2 = FiniteElement("N1curl", cell, 1)

    f = Coefficient(V1)
    g = Coefficient(V2)

    J = Jacobian(domain)
    K = JacobianInverse(domain)
    detJ = JacobianDeterminant(domain)

    fr = ReferenceValue(f)
    gr = ReferenceValue(g)

    # Geometric quantities we need:
    J = Jacobian(domain)
    detJ = JacobianDeterminant(domain)
    Jinv = JacobianInverse(domain)
    i, j = indices(2)
    JinvT = as_tensor(Jinv[i, j], (j, i))

    # Contravariant Piola mapping
    M_hdiv = (1.0/detJ) * J
    # Covariant Piola mapping
    M_hcurl = JinvT

    # Test that simple forms with piola mapped vector elements are mapped correctly
    r, s = indices(2)
    k, l = indices(2)
    mappings = {
        f: as_vector(M_hdiv[r,s]*fr[s], r),
        g: as_vector(M_hcurl[k,l]*gr[l], k),
        }

    check_form_pullback(f**2*dx, mappings)
    check_form_pullback(g**2*ds, mappings)
    check_form_pullback((f**2 + g**2/f**2)*dx, mappings)

def test_apply_function_pullbacks_on_mixed_functions_with_mixed_mappings():
    cell = triangle
    domain = as_domain(cell)

    #V0 = TensorElement("DG", cell, 0, symmetry=True)
    V0 = FiniteElement("DG", cell, 0)
    V1 = FiniteElement("RT", cell, 1)
    V2 = FiniteElement("N1curl", cell, 1)
    W = MixedElement((V0, V1, V2))

    f = Coefficient(W)

    J = Jacobian(domain)
    K = JacobianInverse(domain)
    detJ = JacobianDeterminant(domain)

    fr = ReferenceValue(f)

    # Geometric quantities we need:
    J = Jacobian(domain)
    detJ = JacobianDeterminant(domain)
    Jinv = JacobianInverse(domain)
    i, j = indices(2)
    JinvT = as_tensor(Jinv[i, j], (j, i))

    # Contravariant Piola mapping
    M_hdiv = (1.0/detJ) * J
    # Covariant Piola mapping
    M_hcurl = JinvT

    # Test that simple forms with piola mapped vector elements are mapped correctly
    Md = M_hdiv
    Mc = M_hcurl

    # Big Matrix transform version: # FIXME: Change the implementation of this
    M = as_tensor([ [1.0, 0, 0, 0, 0],
                    [0] + list(Md[0,:]) + [0, 0],
                    [0] + list(Md[1,:]) + [0, 0],
                    [0, 0, 0] + list(Mc[0,:]),
                    [0, 0, 0] + list(Mc[1,:]),
                    #[0, Md[0,0], Md[0,1], 0, 0],
                    #[0, Md[1,0], Md[1,1], 0, 0],
                    #[0, 0, 0, Mc[0,0], Mc[0,1]],
                    #[0, 0, 0, Mc[1,0], Mc[1,1]],
                    ])

    r, s = indices(2)
    mappings = {
        f: as_vector(M[r,s]*fr[s], r),
        }

    check_form_pullback(f**2*dx, mappings)
