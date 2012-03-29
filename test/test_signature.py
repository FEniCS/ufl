#!/usr/bin/env python

"""
Test the computation of form signatures.
"""

# These are thin wrappers on top of unittest.TestCase and unittest.main
from ufltestcase import UflTestCase, main

# This imports everything external code will see from ufl
from ufl import *

#from ufl.classes import *
#from ufl.algorithms import *

def compute_multiindex_hashdata(expr, index_numbering):
    data = []
    for i in expr:
        if isinstance(i, Index):
            j = index_numbering.get(i)
            if i is None:
                # Use negative ints for Index
                j = -len(index_numbering)
                index_numbering[i] = j
            data.append(j)
        else:
            # Use positive ints for FixedIndex
            data.append(int(i))
    return data

def compute_signature(form):
    hashdata = []
    for integral in form.integrals():
        measure = integral.measure()
        integrand = integral.integrand()

        # Build hashdata for all terminals first
        index_numbering = {}
        coefficients = set()
        arguments = set()
        for e in traverse_terminals():
            if isinstance(expr, MultiIndex):
                terminal_hashdata[e] = compute_multiindex_hashdata(expr, index_numbering)
            elif isinstance(expr, Coefficient):
                coefficients.append(expr)
            elif isinstance(expr, Argument):
                arguments.append(expr)
            elif isinstance(expr, Counted):
                error("Not implemented hashing for Counted subtype %s" % type(expr))
            else:
                terminal_hashdata[e] = repr(expr)
        coefficients = sorted(coefficients, key=lambda x: x.count())
        arguments = sorted(arguments, key=lambda x: x.count())
        for i, e in enumerate(coefficients):
            terminal_hashdata[e] = repr(e.reconstruct(count=i))
        for i, e in enumerate(arguments):
            terminal_hashdata[e] = repr(e.reconstruct(count=i))

        # Build hashdata for expression
        expression_hashdata = []
        for expr in fast_pre_traversal(integrand):
            if isinstance(expr, Terminal):
                data = terminal_hashdata[expr]
            else:
                data = expr._uflclassid
            expression_hashdata.append(data)
        integral_hashdata = (repr(measure), expression_hashdata)
        hashdata.append(integral_hashdata)

    return hashlib.sha224(str(hashdata)).hexdigest()

class SignatureTestCase(UflTestCase):

    # TODO: How do we test signature reliably?
    # TODO: Check that all relevant terminal propeties affect the sig
    # TODO: Check that operator types affect the sig
    # TODO: Check that counts do NOT affect the sig
    # TODO: Check that we do not get collisions for large sets of forms
    # TODO: Test multiindex hashdata in particular!

    def test_signature_is_affected_by_element_properties(self):
        count = 0
        sigs = set()
        hashes = set()
        reprs = set()
        for family in ("CG", "DG"):
            for cell in (triangle, tetrahedron, quadrilateral):
                for degree in (1,2):
                    V = FiniteElement(family, cell, degree)
                    u = Coefficient(V)
                    v = TestFunction(V)
                    x = cell.x
                    w = as_vector([v,v])
                    f = dot(w, u*x)
                    a = f*dx

                    sig = a.signature()
                    sigs.add(sig)
                    self.assertTrue(sig)

                    hashes.add(hash(a))
                    reprs.add(repr(a))

                    count += 1

        print '\n'.join(sigs)
        self.assertTrue(len(sigs) == count)
        self.assertTrue(len(reprs) == count)
        self.assertTrue(len(hashes) == count)

# Don't touch these lines, they allow you to run this file directly
if __name__ == "__main__":
    main()

