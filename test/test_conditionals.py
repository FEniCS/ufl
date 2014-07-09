#!/usr/bin/env python

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-20 -- 2012-11-30"

from ufltestcase import UflTestCase, main

from ufl import *
#from ufl.algorithms import *
from ufl.classes import *

element = FiniteElement("Lagrange", triangle, 1)
f = Coefficient(element)
g = Coefficient(element)

class ConditionalsTestCase(UflTestCase):

    def test_conditional_does_not_allow_bool_condition(self):
        # The reason for this test is that it protects from the case
        # conditional(a == b, t, f) in which a == b means comparing representations
        self.assertRaises(UFLException, lambda: conditional(True, 1, 0))

    def test_eq_produces_ufl_expr(self):
        expr1 = eq(f, f)
        expr2 = eq(f, g)
        expr3 = eq(f, g)
        self.assertTrue(isinstance(expr1, EQ))
        self.assertTrue(isinstance(expr2, EQ))
        self.assertFalse(bool(expr1 == expr2))
        self.assertTrue(bool(expr1 != expr2))
        self.assertTrue(bool(expr2 == expr3))

    def test_eq_oper_produces_bool(self):
        expr1 = f == f
        expr2 = f == g
        self.assertIsInstance(expr1, bool)
        self.assertIsInstance(expr2, bool)
        self.assertTrue(expr1)
        self.assertFalse(expr2)

    def xtest_eq_produces_ufl_expr(self):
        expr1 = f == g
        expr2 = eq(f, g)
        self.assertTrue(isinstance(expr1, EQ))
        self.assertTrue(isinstance(expr2, EQ))
        self.assertTrue(bool(expr1 == expr2))
        self.assertFalse(bool(expr1 != expr2))


    def test_eq_produces_ufl_expr(self):
        expr1 = eq(f, g)
        expr2 = eq(f, f)
        expr3 = f == g
        expr4 = f == f
        # Correct types:
        self.assertTrue(isinstance(expr1, EQ))
        self.assertTrue(isinstance(expr2, EQ))
        self.assertTrue(isinstance(expr3, bool))
        self.assertTrue(isinstance(expr4, bool))
        # Comparing representations correctly:
        self.assertTrue(bool(expr1 == eq(f, g)))
        self.assertTrue(bool(expr1 != eq(g, g)))
        self.assertTrue(bool(expr2 == eq(f, f)))
        self.assertTrue(bool(expr2 != eq(g, f)))
        # Bool evaluation yields actual bools:
        self.assertTrue(isinstance(bool(expr1), bool))
        self.assertTrue(isinstance(bool(expr2), bool))
        self.assertFalse(expr3)
        self.assertTrue(expr4)
        # Allow use in boolean python expression context:
        # NB! This means comparing representations! Required by dict and set.
        self.assertFalse(bool(expr1))
        self.assertTrue(bool(expr2))
        self.assertFalse(bool(expr3))
        self.assertTrue(bool(expr4))

    def test_ne_produces_ufl_expr(self):
        expr1 = ne(f, g)
        expr2 = ne(f, f)
        expr3 = f != g
        expr4 = f != f
        # Correct types:
        self.assertTrue(isinstance(expr1, NE))
        self.assertTrue(isinstance(expr2, NE))
        self.assertTrue(isinstance(expr3, bool))
        self.assertTrue(isinstance(expr4, bool))
        # Comparing representations correctly:
        self.assertTrue(bool(expr1 == ne(f, g)))
        self.assertTrue(bool(expr1 != ne(g, g)))
        self.assertTrue(bool(expr2 == ne(f, f)))
        self.assertTrue(bool(expr2 != ne(g, f)))
        self.assertFalse(bool(expr2 == expr3))
        # Bool evaluation yields actual bools:
        self.assertTrue(isinstance(bool(expr1), bool))
        self.assertTrue(isinstance(bool(expr2), bool))
        # Allow use in boolean python expression context:
        # NB! This means the opposite of ==, i.e. comparing representations!
        self.assertTrue(bool(expr1))
        self.assertFalse(bool(expr2))
        self.assertTrue(bool(expr1))
        self.assertFalse(bool(expr2))

    def test_lt_produces_ufl_expr(self):
        expr1 = lt(f, g)
        expr2 = f < g
        # Correct types (no bools here!):
        self.assertTrue(isinstance(expr1, LT))
        self.assertTrue(isinstance(expr2, LT))
        # Representations are the same:
        self.assertTrue(bool(expr1 == expr2))
        # Protection from misuse in boolean python expression context:
        self.assertRaises(UFLException, lambda: bool(expr1))

    def test_gt_produces_ufl_expr(self):
        expr1 = gt(f, g)
        expr2 = f > g
        # Correct types (no bools here!):
        self.assertTrue(isinstance(expr1, GT))
        self.assertTrue(isinstance(expr2, GT))
        # Representations are the same:
        self.assertTrue(bool(expr1 == expr2))
        # Protection from misuse in boolean python expression context:
        self.assertRaises(UFLException, lambda: bool(expr1))


    def test_le_produces_ufl_expr(self):
        expr1 = le(f, g)
        expr2 = f <= g
        # Correct types (no bools here!):
        self.assertTrue(isinstance(expr1, LE))
        self.assertTrue(isinstance(expr2, LE))
        # Representations are the same:
        self.assertTrue(bool(expr1 == expr2))
        # Protection from misuse in boolean python expression context:
        self.assertRaises(UFLException, lambda: bool(expr1))

    def test_ge_produces_ufl_expr(self):
        expr1 = ge(f, g)
        expr2 = f >= g
        # Correct types (no bools here!):
        self.assertTrue(isinstance(expr1, GE))
        self.assertTrue(isinstance(expr2, GE))
        # Representations are the same:
        self.assertTrue(bool(expr1 == expr2))
        # Protection from misuse in boolean python expression context:
        self.assertRaises(UFLException, lambda: bool(expr1))


if __name__ == "__main__":
    main()
