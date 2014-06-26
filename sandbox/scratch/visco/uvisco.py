#!/usr/bin/python
# Copyright (C) 2012 Harish Narayanan

DOLFIN = 0
if DOLFIN:
    from dolfin import *
else:
    from ufl import *

import time
from sys import getsizeof
import ufl.classes
from ufl.algorithms import expand_indices, Graph
from pympler.asizeof import asizeof

#import sfc
#sfc.set_level(DEBUG)

#set_log_level(DEBUG)

if DOLFIN:
    parameters["form_compiler"]["name"] = "sfc"

# parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {
#    "quadrature_degree": 2,
#    "eliminate_zeros": True,
#    "precompute_basis_const": True,
#    "precompute_ip_const": True,
#    "optimize": True,
#    "log_level" : DEBUG,
}

if DOLFIN:
    mesh = UnitCube(1,1,1)
else:
    cell = tetrahedron
gdim = 3

# Reference fibre, sheet and sheet-normal directions
if DOLFIN:
    f0 = Constant((1, 0, 0))
    s0 = Constant((0, 1, 0))
    n0 = Constant((0, 0, 1))
else:
    f0 = VectorConstant(cell)
    s0 = VectorConstant(cell)
    n0 = VectorConstant(cell)

# Material parameters for Figure 7 in HolzapfelOgden2009
if DOLFIN:
    a    =  Constant(0.500)   #kPa
    b    =  Constant(8.023)
    a_f  =  Constant(16.472)  #kPa
    b_f  =  Constant(16.026)
    a_s  =  Constant(2.481)   #kPa
    b_s  =  Constant(11.120)
    a_fs =  Constant(0.356)   #kPa
    b_fs =  Constant(11.436)

    kappa = Constant(2.0e6)   #kPa
    beta  = Constant(9.0)
else:
    a    =  Constant(cell)
    b    =  Constant(cell)
    a_f  =  Constant(cell)
    b_f  =  Constant(cell)
    a_s  =  Constant(cell)
    b_s  =  Constant(cell)
    a_fs =  Constant(cell)
    b_fs =  Constant(cell)

    kappa = Constant(cell)
    beta  = Constant(cell)

# Strain energy functions for the passive myocardium
def psi_iso_inf(I1_bar, I4_f_bar, I4_s_bar, I8_fs_bar, I8_fn_bar):
    return(a/(2*b)*exp(b*(I1_bar - 3)) \
          + a_f/(2*b_f)*(exp(b_f*(I4_f_bar - 1)**2) - 1) \
          + a_s/(2*b_s)*(exp(b_s*(I4_s_bar - 1)**2) - 1) \
          + a_fs/(2*b_fs)*(exp(b_fs*I8_fs_bar**2) - 1))

def psi_vol_inf(J):
    return(kappa*(1/(beta**2)*(beta*ln(J) + 1/(J**beta) - 1)))

# Define the elastic response of the material
def P(u):
    # Kinematics
    I = Identity(gdim)    # Identity tensor
    F = I + grad(u)             # Deformation gradient
    C = F.T*F                   # Right Cauchy-Green tensor
    J = variable(det(F))        # Jacobian
    C_bar = J**(-2.0/3.0)*C     # Modified right Cauchy-Green tensor

    # Principle isotropic invariants
    I1_bar = variable(tr(C_bar))
    I2_bar = variable(0.5*(tr(C_bar)**2 - tr(C_bar*C_bar)))

    # Anisotropic (quasi) invariants
    I4_f_bar = variable(inner(f0, C_bar*f0))
    I4_s_bar = variable(inner(s0, C_bar*s0))
    I8_fs_bar = variable(inner(f0, C_bar*s0))
    I8_fn_bar = variable(inner(f0, C_bar*n0))

    # Strain energy functions
    psi_iso = psi_iso_inf(I1_bar, I4_f_bar, I4_s_bar, I8_fs_bar, I8_fn_bar)
    psi_vol = psi_vol_inf(J)

    # Define the second Piola-Kirchhoff stress in terms of the invariants
    # S_bar =   2*(diff(psi_iso, I1_bar) + diff(psi_iso, I2_bar))*I \
    #         - 2*diff(psi_iso, I2_bar)*C_bar \
    #         + 2*diff(psi_iso, I4_f_bar)*outer(f0, f0) \
    #         + 2*diff(psi_iso, I4_s_bar)*outer(s0, s0) \
    #         + diff(psi_iso, I8_fs_bar)*(outer(f0, s0) + outer(s0, f0)) \
    #         + diff(psi_iso, I8_fn_bar)*(outer(f0, n0) + outer(n0, f0))

    # Hand compute the second Piola-Kirchhoff stress in terms of the invariants
    S_bar = as_matrix([[a*exp(b*(I1_bar - 3)) + a_f*(2*I4_f_bar - 2)*exp(b_f*(I4_f_bar - 1)**2), I8_fs_bar*a_fs*exp(I8_fs_bar**2*b_fs), 0.0],
                       [I8_fs_bar*a_fs*exp(I8_fs_bar**2*b_fs), a*exp(b*(I1_bar - 3)) + a_s*(2*I4_s_bar - 2)*exp(b_s*(I4_s_bar - 1)**2), 0.0],
                       [0.0, 0.0, a*exp(b*(I1_bar - 3))]])


    Dev_S_bar = S_bar - (1.0/3.0)*inner(S_bar, C)*inv(C)
    S_iso_inf = J**(-2.0/3.0)*Dev_S_bar
    S_vol_inf = J*diff(psi_vol, J)*inv(C)

    # Return the first Piola-Kirchhoff stress
    return (F*(S_iso_inf + S_vol_inf))

def build_forms():
    if DOLFIN:
        V = VectorFunctionSpace(mesh, "Lagrange", 1)
        Q = FunctionSpace(mesh, "Lagrange", 1)
        u  = Function(V)                 # Displacement from previous iteration
    else:
        V = VectorElement("Lagrange", tetrahedron, 1)
        Q = FiniteElement("Lagrange", tetrahedron, 1)
        u  = Coefficient(V)                 # Displacement from previous iteration
    du = TrialFunction(V)            # Incremental displacement
    v  = TestFunction(V)             # Test function

    F = inner(P(u), grad(v))*dx
    J = derivative(F, u, du)
    return F, J

def printmem():
    print()
    ufl.expr.print_expr_statistics()
    print()

def find_the_memory_thief(expr):
    reprtot = 0
    reprtot2 = 0
    memuse = {}
    tmemuse = {}
    ids = set()
    for e in ufl.algorithms.pre_traversal(expr):
        if id(e) in ids:
            continue
        ids.add(id(e))
        if hasattr(e, '_repr'):
            r = repr(e)
            reprtot += getsizeof(r)
            reprtot2 += len(r)
        n = type(e).__name__
        memuse[n] = max(memuse.get(n,0), getsizeof(e))
        tmemuse[n] = tmemuse.get(n,0) + getsizeof(e)
    worst = sorted(list(memuse.items()), key=lambda x: x[1])
    tworst = sorted(list(tmemuse.items()), key=lambda x: x[1])
    print()
    print('ids:', len(ids))
    print('reprtot bytes:', reprtot)
    print('reprtot2 len: ', reprtot2)
    print('totmem (MB):', (sum(x[1] for x in tworst) / float(1024**2)))
    print("-"*60, 'worst')
    print(worst)
    print("-"*60, 'tworst')
    print(tworst)
    print()

def print_expr_size(expr):
    print("::", getsizeof(expr), " ", asizeof(expr), " ", \
          sum(1 for _ in ufl.algorithms.pre_traversal(expr)))

msize, mtime = 0, 0
def process_form(F):
    global msize, mtime

    printmem()
    print("size of form", asizeof(F))
    print_expr_size(F.integrals(Measure.CELL)[0].integrand())

    t1 = time.time()
    print('starting preprocess')
    Ffd = F.compute_form_data()
    t2 = time.time()
    print('preprocess took %d s' % (t2-t1))
    print("size of form data (in MB)", asizeof(Ffd)/float(1024**2))
    print_expr_size(Ffd.preprocessed_form.integrals(Measure.CELL)[0].integrand())
    printmem()

    printmem()
    t1 = time.time()
    print('starting expand_indices')
    eiF = expand_indices(Ffd.preprocessed_form)
    t2 = time.time()
    print('expand_indices took %d s' % (t2-t1))
    #print "REPR LEN", len(repr(eiF))
    msize = asizeof(eiF)/float(1024**2)
    mtime = t2-t1
    print("size of expanded form (in MB)", msize)
    print_expr_size(eiF.integrals(Measure.CELL)[0].integrand())
    printmem()

    t1 = time.time()
    print('starting graph building')
    FG = Graph(eiF.integrals(Measure.CELL)[0].integrand())
    t2 = time.time()
    print('graph building took %d s' % (t2-t1))
    print("size of graph (in MB)", asizeof(FG)/float(1024**2))
    printmem()

    return eiF

def main():
    F, J = build_forms()

    if 0:
        print('\n', '='*50, 'F')
        ei = process_form(F)

    if 1:
        print('\n', '='*50, 'J')
        ei = process_form(J)

    print('\n', '='*50, 'mem of only eiJ')
    del F
    del J
    printmem()

    find_the_memory_thief(ei.integrals(Measure.CELL)[0].integrand())

    #print formatted_analysis(ei, classes=True)

    print()
    print(msize)
    print(mtime)

    try:
        from guppy import hpy
        hp = hpy()
        print("heap:")
        print(hp.heap())
    except:
        print("No guppy installed!")

main()

