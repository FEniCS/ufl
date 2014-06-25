#!/usr/bin/env python
"""
To avoid typing errors, this code was used to generate expressions
for the expansion of compound expressions like the cofactor, inverse,
and deviatoric part of a matrix.
"""

import re
from swiginac import *

def matrix2Matrix(A):
    s = str(A)
    res = re.subn(r"A(.)(.)", r"A[\1,\2]", s)
    return "Matrix(%s)" % res[0]

def cofac(A):
    return A.determinant() * A.inverse().transpose()

def cofacstr(d):
    A = symbolic_matrix(d, d, "A")
    return matrix2Matrix(cofac(A))

def dev(A):
    d = A.rows()
    I = matrix(d, d)
    for i in range(d):
        I[i,i] = 1.0
    alpha = sum(A[i,i] for i in range(d))
    return A - alpha/d*I

def devstr(d):
    A = symbolic_matrix(d, d, "A")
    return re.sub(r"(.)/", r"\1./", matrix2Matrix(dev(A)))

print("Cofactors:")
print()
print(cofacstr(2))
print()
print(cofacstr(3))
print()
print(cofacstr(4))
print()

print("Deviatoric:")
print()
print(devstr(2))
print()
print(devstr(3))
print()

