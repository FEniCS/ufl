#!/usr/bin/env python

__authors__ = "Martin Sandve Alnes"
__date__ = "2009-03-19 -- 2009-03-24"

# Modified by Anders Logg, 2008
# Modified by Garth N. Wells, 2009

from ufltestcase import UflTestCase, main
import math
from pprint import *

from ufl import *
from ufl.algorithms import * 
from ufl.classes import Sum, Product

# TODO: add more tests, covering all utility algorithms

class ExpandIndicesTestCase(UflTestCase):

    def setUp(self):
        cell = triangle
        element = FiniteElement("Lagrange", cell, 1)
        velement = VectorElement("Lagrange", cell, 1)
        telement = TensorElement("Lagrange", cell, 1)
        self.sf = Coefficient(element)
        self.sf2 = Coefficient(element)
        self.vf = Coefficient(velement)
        self.tf = Coefficient(telement)
        
        # Note: the derivatives of these functions make no sense, but
        #       their unique constant values are used for validation.
        
        def SF(x, derivatives=()): 
            if derivatives == ():
                return 3
            elif derivatives == (0,):
                return 0.30
            elif derivatives == (1,):
                return 0.31
            return 0
        
        def SF2(x, derivatives=()):
            if derivatives == ():
                return 3
            elif derivatives == (0,):
                return 0.30
            elif derivatives == (1,):
                return 0.31
            elif derivatives == (0,0):
                return 3.300
            elif derivatives in ((1,0), (0,1)):
                return 3.310
            elif derivatives == (1,1):
                return 3.311
            return 0
        
        def VF(x, derivatives=()):
            if derivatives == ():
                return (5, 7)
            elif derivatives == (0,):
                return (0.50, 0.70)
            elif derivatives == (1,):
                return (0.51, 0.71)
            return (0, 0)
        
        def TF(x, derivatives=()):
            if derivatives == ():
                return ((11, 13), (17, 19))
            elif derivatives == (0,):
                return ((1.10, 1.30), (1.70, 1.90))
            elif derivatives == (1,):
                return ((1.11, 1.31), (1.71, 1.91))
            return ((0, 0), (0, 0))
        
        self.x = (1.23, 3.14)
        self.mapping = { self.sf: SF, self.sf2: SF2, self.vf: VF, self.tf: TF }
        
    def compare(self, f, value):
        g = expand_derivatives(f)
        gv = g(self.x, self.mapping)
        self.assertAlmostEqual(gv, value)

        g = expand_indices(g)
        gv = g(self.x, self.mapping)
        self.assertAlmostEqual(gv, value)

        g = renumber_indices(g)
        gv = g(self.x, self.mapping)
        self.assertAlmostEqual(gv, value)

    def test_basic_expand_indices(self):
        sf = self.sf
        vf = self.vf
        tf = self.tf
        compare = self.compare

        # Simple expressions with no indices or derivatives to expand
        compare(sf, 3)
        compare(sf + 1, 4)
        compare(sf - 2.5, 0.5)
        compare(sf/2, 1.5)
        compare(sf/0.5, 6)
        compare(sf**2, 9)
        compare(sf**0.5, 3**0.5)
        compare(sf**3, 27)
        compare(0.5**sf, 0.5**3)
        compare(sf * (sf/6), 1.5)
        compare(sin(sf), math.sin(3))
        compare(cos(sf), math.cos(3))
        compare(exp(sf), math.exp(3))
        compare(ln(sf), math.log(3))

        # Simple indexing
        compare(vf[0], 5)
        compare(vf[0] + 1, 6)
        compare(vf[0] - 2.5, 2.5)
        compare(vf[0]/2, 2.5)
        compare(vf[0]/0.5, 10)
        compare(vf[0]**2, 25)
        compare(vf[0]**0.5, 5**0.5)
        compare(vf[0]**3, 125)
        compare(0.5**vf[0], 0.5**5)
        compare(vf[0] * (vf[0]/6), 5*(5./6))
        compare(sin(vf[0]), math.sin(5))
        compare(cos(vf[0]), math.cos(5))
        compare(exp(vf[0]), math.exp(5))
        compare(ln(vf[0]), math.log(5))
        
        # Double indexing
        compare(tf[1,1], 19)
        compare(tf[1,1] + 1, 20)
        compare(tf[1,1] - 2.5, 16.5)
        compare(tf[1,1]/2, 9.5)
        compare(tf[1,1]/0.5, 38)
        compare(tf[1,1]**2, 19**2)
        compare(tf[1,1]**0.5, 19**0.5)
        compare(tf[1,1]**3, 19**3)
        compare(0.5**tf[1,1], 0.5**19)
        compare(tf[1,1] * (tf[1,1]/6), 19*(19./6))
        compare(sin(tf[1,1]), math.sin(19))
        compare(cos(tf[1,1]), math.cos(19))
        compare(exp(tf[1,1]), math.exp(19))
        compare(ln(tf[1,1]), math.log(19))

    def test_expand_indices_index_sum(self):
        sf = self.sf
        vf = self.vf
        tf = self.tf
        compare = self.compare

        # Basic index sums
        compare(vf[i]*vf[i], 5*5+7*7)
        compare(vf[j]*tf[i,j]*vf[i], 5*5*11 + 5*7*13 + 5*7*17 + 7*7*19)
        compare(vf[j]*tf.T[j,i]*vf[i], 5*5*11 + 5*7*13 + 5*7*17 + 7*7*19)
        compare(tf[i,i], 11 + 19)
        compare(tf[i,j]*(tf[j,i]+outer(vf, vf)[i,j]), (5*5+11)*11 + (7*5+17)*13 + (7*5+13)*17 + (7*7+19)*19)
        compare( as_tensor( as_tensor(tf[i,j], (i,j))[k,l], (l,k) )[i,i], 11 + 19 )
    
    def test_expand_indices_derivatives(self):
        sf = self.sf
        vf = self.vf
        tf = self.tf
        compare = self.compare

        # Basic derivatives
        compare(sf.dx(0), 0.3)
        compare(sf.dx(1), 0.31)
        compare(sf.dx(i)*vf[i], 0.30*5 + 0.31*7)
        compare(vf[j].dx(i)*vf[i].dx(j), 0.50*0.50 + 0.51*0.70 + 0.70*0.51 + 0.71*0.71)

    def test_expand_indices_hyperelasticity(self):
        sf = self.sf
        vf = self.vf
        tf = self.tf
        compare = self.compare

        # Deformation gradient
        I = Identity(2)
        u = vf
        F = I + grad(u)
        # F = (1 + vf[0].dx(0), vf[0].dx(1), vf[1].dx(0), 1 + vf[1].dx(1))
        # F = (1 + 0.50,        0.51,        0.70,        1 + 0.71)
        F00 = 1 + 0.50
        F01 = 0.51
        F10 = 0.70
        F11 = 1 + 0.71
        compare(F[0,0], F00)
        compare(F[0,1], F01)
        compare(F[1,0], F10)
        compare(F[1,1], F11)

        J = det(F)
        compare(J, (1 + 0.50)*(1 + 0.71) - 0.70*0.51)
        
        # Strain tensors
        C = F.T*F
        # Cij = sum_k Fki Fkj
        C00 = F00*F00 + F10*F10
        C01 = F00*F01 + F10*F11
        C10 = F01*F00 + F11*F10
        C11 = F01*F01 + F11*F11
        compare(C[0,0], C00)
        compare(C[0,1], C01)
        compare(C[1,0], C10)
        compare(C[1,1], C11)
        
        E = (C-I)/2
        E00 = (C00-1)/2
        E01 = (C01  )/2
        E10 = (C10  )/2
        E11 = (C11-1)/2
        compare(E[0,0], E00)
        compare(E[0,1], E01)
        compare(E[1,0], E10)
        compare(E[1,1], E11)
        
        # Strain energy
        Q = inner(E, E)
        Qvalue = E00**2 + E01**2 + E10**2 + E11**2
        compare(Q, Qvalue)

        K = 0.5
        psi = (K/2)*exp(Q)
        compare(psi, 0.25*math.exp(Qvalue))

    def test_expand_indices_div_grad(self):
        sf = self.sf
        sf2 = self.sf2
        vf = self.vf
        tf = self.tf
        compare = self.compare

        a = div(grad(sf))
        compare(a, 0)

        a = div(grad(sf2))
        compare(a, 3.300 + 3.311)

tests = [ExpandIndicesTestCase]

if __name__ == "__main__":
    main()
