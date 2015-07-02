#!/usr/bin/env py.test

from pytest import raises
from ufl import *
from ufl.algorithms.apply_function_pullbacks import apply_function_pullbacks
from ufl.algorithms import renumber_indices
from ufl.classes import Jacobian, JacobianInverse, JacobianDeterminant, ReferenceValue, CellOrientation

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


def sub_elements_with_mappings(element):
    "Return an ordered list of the largest subelements that have a defined mapping."
    if element.mapping() != "undefined":
        return [element]
    elements = []
    for subelm in element.sub_elements():
        if subelm.mapping() != "undefined":
            elements.append(subelm)
        else:
            elements.extend(sub_elements_with_mappings(subelm))
    return elements

def create_nested_lists(shape):
    if len(shape) == 0:
        return [None]
    elif len(shape) == 1:
        return [None]*shape[0]
    else:
        return [create_nested_lists(shape[1:]) for i in range(shape[0])]

def reshape_to_nested_list(components, shape):
    if len(shape) == 0:
        assert len(components) == 1
        return [components[0]]
    elif len(shape) == 1:
        assert len(components) == shape[0]
        return components
    else:
        n = product(shape[1:])
        return [reshape_to_nested_list(components[n*i:n*(i+1)], shape[1:]) for i in range(shape[0])]

def apply_single_function_pullbacks(g):
    element = g.element()
    mapping = element.mapping()
    domain = g.domain()
    r = ReferenceValue(g)

    gsh = g.ufl_shape
    rsh = r.ufl_shape
    gsize = product(gsh)
    rsize = product(rsh)

    # Create some geometric objects for reuse
    J = Jacobian(domain)
    detJ = JacobianDeterminant(domain)
    Jinv = JacobianInverse(domain)

    tdim = domain.topological_dimension()
    gdim = domain.geometric_dimension()

    # Create contravariant transform for reuse
    transform_hdiv = (1.0/detJ) * J
    if tdim != gdim:
        # Only insert symbolic CellOrientation if tdim != gdim
        transform_hdiv = CellOrientation(domain) * transform_hdiv


    # Shortcut simple cases for a more efficient representation,
    # including directly Piola-mapped elements and mixed elements
    # of any combination of affinely mapped elements without symmetries
    if mapping == "identity":
        assert rsh == gsh
        return r
    elif mapping == "symmetries":
        fcm = element.flattened_sub_element_mapping()
        assert gsize >= rsize
        assert len(fcm) == gsize
        assert sorted(set(fcm)) == sorted(range(rsize))
        g_components = [r[fcm[i]] for i in range(gsize)]
        g_components = reshape_to_nested_list(g_components, gsh)
        f = as_tensor(g_components)
        assert f.ufl_shape == g.ufl_shape
        return f
    elif mapping == "contravariant Piola":
        assert transform_hdiv.ufl_shape == (gsize, rsize)
        i, j = indices(2)
        f = as_vector(transform_hdiv[i, j]*r[j], i)
        #f = as_tensor(transform_hdiv[i, j]*r[k,j], (k,i)) # FIXME: Handle Vector(Piola) here?
        assert f.ufl_shape == g.ufl_shape
        return f
    elif mapping == "covariant Piola":
        assert Jinv.ufl_shape == (rsize, gsize)
        i, j = indices(2)
        f = as_vector(Jinv[j, i]*r[j], i)
        #f = as_tensor(Jinv[j, i]*r[k,j], (k,i)) # FIXME: Handle Vector(Piola) here?
        assert f.ufl_shape == g.ufl_shape
        return f


    # By placing components in a list and using as_vector at the end, we're
    # assuming below that both global function g and its reference value r
    # have vector shape, which is the case for most elements with the exceptions:
    # - TensorElements
    #   - All cases with scalar subelements and without symmetries are covered by the shortcut above
    #     (ONLY IF REFERENCE VALUE SHAPE PRESERVES TENSOR RANK)
    #   - All cases with scalar subelements and without symmetries are covered by the shortcut above
    # - VectorElements of vector-valued basic elements (FIXME)
    # - TensorElements with symmetries (FIXME)
    # - Tensor-valued FiniteElements (the new Regge elements)
    assert len(gsh) == 1
    assert len(rsh) == 1

    g_components = [None]*gsize
    gpos = 0
    rpos = 0
    for subelm in sub_elements_with_mappings(element):
        gm = product(subelm.value_shape())
        rm = product(subelm.reference_value_shape())

        mp = subelm.mapping()
        if mp == "identity":
            assert gm == rm
            for i in range(gm):
                g_components[gpos + i] = r[rpos + i]

        elif mp == "symmetries":
            """
            tensor_element.value_shape() == (2,2)
            tensor_element.reference_value_shape() == (3,)
            tensor_element.symmetry() == { (1,0): (0,1) }
            tensor_element.component_mapping() == { (0,0): 0, (0,1): 1, (1,0): 1, (1,1): 2 }
            tensor_element.flattened_component_mapping() == { 0: 0, 1: 1, 2: 1, 3: 2 }
            """
            fcm = subelm.flattened_sub_element_mapping()
            assert gm >= rm
            assert len(fcm) == gm
            assert sorted(set(fcm)) == sorted(range(rm))
            for i in range(gm):
                g_components[gpos + i] = r[rpos + fcm[i]]

        elif mp == "contravariant Piola":
            assert transform_hdiv.ufl_shape == (gm, rm)
            # Get reference value vector corresponding to this subelement:
            rv = as_vector([r[rpos+k] for k in range(rm)])
            # Apply transform with IndexSum over j for each row
            j = Index()
            for i in range(gm):
                g_components[gpos + i] = transform_hdiv[i, j]*rv[j]

        elif mp == "covariant Piola":
            assert Jinv.ufl_shape == (rm, gm)
            # Get reference value vector corresponding to this subelement:
            rv = as_vector([r[rpos+k] for k in range(rm)])
            # Apply transform with IndexSum over j for each row
            j = Index()
            for i in range(gm):
                g_components[gpos + i] = Jinv[j, i]*rv[j]

        else:
            error("Unknown subelement mapping type %s for element %s." % (mp, str(subelm)))

        gpos += gm
        rpos += rm

    # Wrap up components in a vector, must return same shape as input function g
    assert len(gsh) == 1
    f = as_vector(g_components)
    assert f.ufl_shape == g.ufl_shape
    return f


def check_single_function_pullback(g, mappings):
    expected = mappings[g]
    actual = apply_single_function_pullbacks(g)
    rexp = renumber_indices(expected)
    ract = renumber_indices(actual)
    if not rexp == ract:
        print
        print "In check_single_function_pullback:"
        print "input:"
        print repr(g)
        print "expected:"
        print str(rexp)
        print "actual:"
        print str(ract)
        print "signatures:"
        print (expected**2*dx).signature()
        print (actual**2*dx).signature()
        print
    assert ract == rexp


def test_apply_single_function_pullbacks_triangle3d():
    triangle3d = Cell("triangle", geometric_dimension=3)
    cell = triangle3d
    domain = as_domain(cell)

    U = FiniteElement("CG", cell, 1)
    V = VectorElement("CG", cell, 1)
    Vd = FiniteElement("RT", cell, 1)
    Vc = FiniteElement("N1curl", cell, 1)
    T = TensorElement("CG", cell, 1)
    S = TensorElement("CG", cell, 1, symmetry=True)

    Um = U*U
    Vm = U*V
    Vdm = V*Vd
    Vcm = Vd*Vc
    Tm = Vc*T
    Sm = T*S

    W = S*T*Vc*Vd*V*U

    u = Coefficient(U)
    v = Coefficient(V)
    vd = Coefficient(Vd)
    vc = Coefficient(Vc)
    t = Coefficient(T)
    s = Coefficient(S)

    um = Coefficient(Um)
    vm = Coefficient(Vm)
    vdm = Coefficient(Vdm)
    vcm = Coefficient(Vcm)
    tm = Coefficient(Tm)
    sm = Coefficient(Sm)

    w = Coefficient(W)

    ru = ReferenceValue(u)
    rv = ReferenceValue(v)
    rvd = ReferenceValue(vd)
    rvc = ReferenceValue(vc)
    rt = ReferenceValue(t)
    rs = ReferenceValue(s)

    rum = ReferenceValue(um)
    rvm = ReferenceValue(vm)
    rvdm = ReferenceValue(vdm)
    rvcm = ReferenceValue(vcm)
    rtm = ReferenceValue(tm)
    rsm = ReferenceValue(sm)

    rw = ReferenceValue(w)
    assert len(w) == 9 + 9 + 3 + 3 + 3 + 1
    assert len(rw) == 6 + 9 + 2 + 2 + 3 + 1
    assert len(w) == 28
    assert len(rw) == 23

    # Geometric quantities we need:
    J = Jacobian(domain)
    detJ = JacobianDeterminant(domain)
    Jinv = JacobianInverse(domain)
    o = CellOrientation(domain)
    i, j, k, l = indices(4)

    # Contravariant H(div) Piola mapping:
    M_hdiv = o * ((1.0/detJ) * J)
    # Covariant H(curl) Piola mapping: Jinv.T

    mappings = {
        # Simple elements should get a simple representation
        u: ru,
        v: rv,
        vd: as_vector(M_hdiv[i,j]*rvd[j], i),
        vc: as_vector(Jinv[j,i]*rvc[j], i),
        t: rt,
        s: as_tensor([[rs[0], rs[1], rs[2]],
                      [rs[1], rs[3], rs[4]],
                      [rs[2], rs[4], rs[5]]]),
        # Mixed elements become a bit more complicated
        um: rum,
        vm: rvm,
        vdm: as_vector([
            # V
            rvdm[0],
            rvdm[1],
            rvdm[2],
            # Vd
            M_hdiv[0,j]*as_vector([rvdm[3], rvdm[4]])[j],
            M_hdiv[1,j]*as_vector([rvdm[3], rvdm[4]])[j],
            M_hdiv[2,j]*as_vector([rvdm[3], rvdm[4]])[j],
            ]),
        vcm: as_vector([
            # Vd
            M_hdiv[0,j]*as_vector([rvcm[0], rvcm[1]])[j],
            M_hdiv[1,j]*as_vector([rvcm[0], rvcm[1]])[j],
            M_hdiv[2,j]*as_vector([rvcm[0], rvcm[1]])[j],
            # Vc
            Jinv[i,0]*as_vector([rvcm[2], rvcm[3]])[i],
            Jinv[i,1]*as_vector([rvcm[2], rvcm[3]])[i],
            Jinv[i,2]*as_vector([rvcm[2], rvcm[3]])[i],
            ]),
        tm: as_vector([
            # Vc
            Jinv[i,0]*as_vector([rtm[0], rtm[1]])[i],
            Jinv[i,1]*as_vector([rtm[0], rtm[1]])[i],
            Jinv[i,2]*as_vector([rtm[0], rtm[1]])[i],
            # T
            rtm[2], rtm[3], rtm[4],
            rtm[5], rtm[6], rtm[7],
            rtm[8], rtm[9], rtm[10],
            ]),
        sm: as_vector([
            # T
            rsm[0], rsm[1], rsm[2],
            rsm[3], rsm[4], rsm[5],
            rsm[6], rsm[7], rsm[8],
            # S
            rsm[ 9], rsm[10], rsm[11],
            rsm[10], rsm[12], rsm[13],
            rsm[11], rsm[13], rsm[14],
            ]),
        # This combines it all:
        w: as_vector([
            # S
            rw[0], rw[1], rw[2],
            rw[1], rw[3], rw[4],
            rw[2], rw[4], rw[5],
            # T
            rw[6], rw[7], rw[8],
            rw[9], rw[10], rw[11],
            rw[12], rw[13], rw[14],
            # Vc
            Jinv[i,0]*as_vector([rw[15], rw[16]])[i],
            Jinv[i,1]*as_vector([rw[15], rw[16]])[i],
            Jinv[i,2]*as_vector([rw[15], rw[16]])[i],
            # Vd
            M_hdiv[0,j]*as_vector([rw[17], rw[18]])[j],
            M_hdiv[1,j]*as_vector([rw[17], rw[18]])[j],
            M_hdiv[2,j]*as_vector([rw[17], rw[18]])[j],
            # V
            rw[19],
            rw[20],
            rw[21],
            # U
            rw[22],
            ]),
        }

    # Check functions of various elements outside a mixed context
    check_single_function_pullback(u, mappings)
    check_single_function_pullback(v, mappings)
    check_single_function_pullback(vd, mappings)
    check_single_function_pullback(vc, mappings)
    check_single_function_pullback(t, mappings)
    check_single_function_pullback(s, mappings)

    # Check functions of various elements inside a mixed context
    check_single_function_pullback(um, mappings)
    check_single_function_pullback(vm, mappings)
    check_single_function_pullback(vdm, mappings)
    check_single_function_pullback(vcm, mappings)
    check_single_function_pullback(tm, mappings)
    check_single_function_pullback(sm, mappings)

    # Check the ridiculous mixed element W combining it all
    check_single_function_pullback(w, mappings)


def test_apply_single_function_pullbacks_triangle():
    cell = triangle
    domain = as_domain(cell)

    U = FiniteElement("CG", cell, 1)
    V = VectorElement("CG", cell, 1)
    Vd = FiniteElement("RT", cell, 1)
    Vc = FiniteElement("N1curl", cell, 1)
    T = TensorElement("CG", cell, 1)
    S = TensorElement("CG", cell, 1, symmetry=True)

    Um = U*U
    Vm = U*V
    Vdm = V*Vd
    Vcm = Vd*Vc
    Tm = Vc*T
    Sm = T*S

    W = S*T*Vc*Vd*V*U

    u = Coefficient(U)
    v = Coefficient(V)
    vd = Coefficient(Vd)
    vc = Coefficient(Vc)
    t = Coefficient(T)
    s = Coefficient(S)

    um = Coefficient(Um)
    vm = Coefficient(Vm)
    vdm = Coefficient(Vdm)
    vcm = Coefficient(Vcm)
    tm = Coefficient(Tm)
    sm = Coefficient(Sm)

    w = Coefficient(W)

    ru = ReferenceValue(u)
    rv = ReferenceValue(v)
    rvd = ReferenceValue(vd)
    rvc = ReferenceValue(vc)
    rt = ReferenceValue(t)
    rs = ReferenceValue(s)

    rum = ReferenceValue(um)
    rvm = ReferenceValue(vm)
    rvdm = ReferenceValue(vdm)
    rvcm = ReferenceValue(vcm)
    rtm = ReferenceValue(tm)
    rsm = ReferenceValue(sm)

    rw = ReferenceValue(w)

    assert len(w) == 4 + 4 + 2 + 2 + 2 + 1
    assert len(rw) == 3 + 4 + 2 + 2 + 2 + 1
    assert len(w) == 15
    assert len(rw) == 14

    # Geometric quantities we need:
    J = Jacobian(domain)
    detJ = JacobianDeterminant(domain)
    Jinv = JacobianInverse(domain)
    i, j, k, l = indices(4)

    # Contravariant H(div) Piola mapping:
    M_hdiv = (1.0/detJ) * J
    # Covariant H(curl) Piola mapping: Jinv.T

    mappings = {
        # Simple elements should get a simple representation
        u: ru,
        v: rv,
        vd: as_vector(M_hdiv[i,j]*rvd[j], i),
        vc: as_vector(Jinv[j,i]*rvc[j], i),
        t: rt,
        s: as_tensor([[rs[0], rs[1]], [rs[1], rs[2]]]),
        # Mixed elements become a bit more complicated
        um: rum,
        vm: rvm,
        vdm: as_vector([
            # V
            rvdm[0],
            rvdm[1],
            # Vd
            M_hdiv[0,j]*as_vector([rvdm[2], rvdm[3]])[j],
            M_hdiv[1,j]*as_vector([rvdm[2], rvdm[3]])[j],
            ]),
        vcm: as_vector([
            # Vd
            M_hdiv[0,j]*as_vector([rvcm[0], rvcm[1]])[j],
            M_hdiv[1,j]*as_vector([rvcm[0], rvcm[1]])[j],
            # Vc
            Jinv[i,0]*as_vector([rvcm[2], rvcm[3]])[i],
            Jinv[i,1]*as_vector([rvcm[2], rvcm[3]])[i],
            ]),
        tm: as_vector([
            # Vc
            Jinv[i,0]*as_vector([rtm[0], rtm[1]])[i],
            Jinv[i,1]*as_vector([rtm[0], rtm[1]])[i],
            # T
            rtm[2], rtm[3],
            rtm[4], rtm[5],
            ]),
        sm: as_vector([
            # T
            rsm[0], rsm[1],
            rsm[2], rsm[3],
            # S
            rsm[4], rsm[5],
            rsm[5], rsm[6],
            ]),
        # This combines it all:
        w: as_vector([
            # S
            rw[0], rw[1],
            rw[1], rw[2],
            # T
            rw[3], rw[4],
            rw[5], rw[6],
            # Vc
            Jinv[i,0]*as_vector([rw[7], rw[8]])[i],
            Jinv[i,1]*as_vector([rw[7], rw[8]])[i],
            # Vd
            M_hdiv[0,j]*as_vector([rw[9], rw[10]])[j],
            M_hdiv[1,j]*as_vector([rw[9], rw[10]])[j],
            # V
            rw[11],
            rw[12],
            # U
            rw[13],
            ]),
        }

    # Check functions of various elements outside a mixed context
    check_single_function_pullback(u, mappings)
    check_single_function_pullback(v, mappings)
    check_single_function_pullback(vd, mappings)
    check_single_function_pullback(vc, mappings)
    check_single_function_pullback(t, mappings)
    check_single_function_pullback(s, mappings)

    # Check functions of various elements inside a mixed context
    check_single_function_pullback(um, mappings)
    check_single_function_pullback(vm, mappings)
    check_single_function_pullback(vdm, mappings)
    check_single_function_pullback(vcm, mappings)
    check_single_function_pullback(tm, mappings)
    check_single_function_pullback(sm, mappings)

    # Check the ridiculous mixed element W combining it all
    check_single_function_pullback(w, mappings)
