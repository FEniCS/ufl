"""Functions implementing compound expressions as equivalent representations using basic operators."""

# Copyright (C) 2008-2014 Martin Sandve Alnes and Anders Logg
#
# This file is part of UFL.
#
# UFL is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# UFL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UFL. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg, 2009-2010

from ufl.log import error, warning
from ufl.assertions import ufl_assert
from ufl.indexing import indices
from ufl.tensors import as_tensor, as_matrix, as_vector
from ufl.operators import sqrt


# Note: To avoid typing errors, the expressions for cofactor and
# deviatoric parts below were created with the script
# tensoralgebrastrings.py under sandbox/scripts/


# Note: Avoiding or delaying application of these horrible expressions
# would be a major improvement to UFL and the form compiler toolchain.
# It could easily be a moderate to major undertaking to get rid of though.


def cross_expr(a, b):
    assert len(a) == 3
    assert len(b) == 3
    def c(i, j):
        return a[i]*b[j] - a[j]*b[i]
    return as_vector((c(1, 2), c(2, 0), c(0, 1)))


def pseudo_determinant_expr(A):
    """Compute the pseudo-determinant of A: sqrt(det(A.T*A))."""
    i, j, k = indices(3)
    ATA = as_tensor(A[k, i]*A[k, j], (i, j))
    return sqrt(determinant_expr(ATA))


def pseudo_inverse_expr(A):
    """Compute the Penrose-Moore pseudo-inverse of A: (A.T*A)^-1 * A.T."""
    i, j, k = indices(3)
    ATA = as_tensor(A[k, i]*A[k, j], (i, j))
    ATAinv = inverse_expr(ATA)
    q, r, s = indices(3)
    return as_tensor(ATAinv[r, q] * A[s, q], (r, s))


def determinant_expr(A):
    "Compute the determinant of A."
    sh = A.shape()
    if sh == ():
        return A
    elif sh[0] == sh[1]:
        if sh[0] == 1:
            return A[0, 0]
        elif sh[0] == 2:
            return determinant_expr_2x2(A)
        elif sh[0] == 3:
            return determinant_expr_3x3(A)
    else:
        return pseudo_determinant_expr(A)

    # TODO: Implement generally for all dimensions?
    error("determinant_expr not implemented for shape %s." % (sh,))

def _det_2x2(B, i, j, k, l):
    return B[i, k]*B[j, l] - B[i, l]*B[j, k]

def determinant_expr_2x2(B):
    return _det_2x2(B, 0, 1, 0, 1)

def determinant_expr_3x3(A):
    return (A[0, 0]*_det_2x2(A, 1, 2, 1, 2) +
            A[0, 1]*_det_2x2(A, 1, 2, 2, 0) +
            A[0, 2]*_det_2x2(A, 1, 2, 0, 1))


def inverse_expr(A):
    "Compute the inverse of A."
    sh = A.shape()
    if sh == ():
        return 1.0 / A
    elif sh[0] == sh[1]:
        if sh[0] == 1:
            return as_tensor(((1.0 / A[0, 0],),))
        else:
            return adj_expr(A) / determinant_expr(A)
    else:
        return pseudo_inverse_expr(A)

def adj_expr(A):
    sh = A.shape()
    ufl_assert(sh[0] == sh[1], "Expecting square matrix.")

    if sh[0] == 2:
        return adj_expr_2x2(A)
    elif sh[0] == 3:
        return adj_expr_3x3(A)
    elif sh[0] == 4:
        return adj_expr_4x4(A)

    error("adj_expr not implemented for dimension %s." % sh[0])

def adj_expr_2x2(A):
    return as_matrix([[A[1, 1], -A[0, 1]],
                      [-A[1, 0], A[0, 0]]])

def adj_expr_3x3(A):
    return as_matrix([
        [ A[2, 2]*A[1, 1] - A[1, 2]*A[2, 1],   -A[0, 1]*A[2, 2] + A[0, 2]*A[2, 1],   A[0, 1]*A[1, 2] - A[0, 2]*A[1, 1]],
        [-A[2, 2]*A[1, 0] + A[1, 2]*A[2, 0],   -A[0, 2]*A[2, 0] + A[2, 2]*A[0, 0],   A[0, 2]*A[1, 0] - A[1, 2]*A[0, 0]],
        [ A[1, 0]*A[2, 1] - A[2, 0]*A[1, 1],    A[0, 1]*A[2, 0] - A[0, 0]*A[2, 1],   A[0, 0]*A[1, 1] - A[0, 1]*A[1, 0]],
        ])

def adj_expr_4x4(A):
    return as_matrix([
        [-A[3, 3]*A[2, 1]*A[1, 2] + A[1, 2]*A[3, 1]*A[2, 3] + A[1, 1]*A[3, 3]*A[2, 2] - A[3, 1]*A[2, 2]*A[1, 3] + A[2, 1]*A[1, 3]*A[3, 2] - A[1, 1]*A[3, 2]*A[2, 3],
         -A[3, 1]*A[0, 2]*A[2, 3] + A[0, 1]*A[3, 2]*A[2, 3] - A[0, 3]*A[2, 1]*A[3, 2] + A[3, 3]*A[2, 1]*A[0, 2] - A[3, 3]*A[0, 1]*A[2, 2] + A[0, 3]*A[3, 1]*A[2, 2],
          A[3, 1]*A[1, 3]*A[0, 2] + A[1, 1]*A[0, 3]*A[3, 2] - A[0, 3]*A[1, 2]*A[3, 1] - A[0, 1]*A[1, 3]*A[3, 2] + A[3, 3]*A[1, 2]*A[0, 1] - A[1, 1]*A[3, 3]*A[0, 2],
          A[1, 1]*A[0, 2]*A[2, 3] - A[2, 1]*A[1, 3]*A[0, 2] + A[0, 3]*A[2, 1]*A[1, 2] - A[1, 2]*A[0, 1]*A[2, 3] - A[1, 1]*A[0, 3]*A[2, 2] + A[0, 1]*A[2, 2]*A[1, 3]],
        [ A[3, 3]*A[1, 2]*A[2, 0] - A[3, 0]*A[1, 2]*A[2, 3] + A[1, 0]*A[3, 2]*A[2, 3] - A[3, 3]*A[1, 0]*A[2, 2] - A[1, 3]*A[3, 2]*A[2, 0] + A[3, 0]*A[2, 2]*A[1, 3],
          A[0, 3]*A[3, 2]*A[2, 0] - A[0, 3]*A[3, 0]*A[2, 2] + A[3, 3]*A[0, 0]*A[2, 2] + A[3, 0]*A[0, 2]*A[2, 3] - A[0, 0]*A[3, 2]*A[2, 3] - A[3, 3]*A[0, 2]*A[2, 0],
         -A[3, 3]*A[0, 0]*A[1, 2] + A[0, 0]*A[1, 3]*A[3, 2] - A[3, 0]*A[1, 3]*A[0, 2] + A[3, 3]*A[1, 0]*A[0, 2] + A[0, 3]*A[3, 0]*A[1, 2] - A[0, 3]*A[1, 0]*A[3, 2],
          A[0, 3]*A[1, 0]*A[2, 2] + A[1, 3]*A[0, 2]*A[2, 0] - A[0, 0]*A[2, 2]*A[1, 3] - A[0, 3]*A[1, 2]*A[2, 0] + A[0, 0]*A[1, 2]*A[2, 3] - A[1, 0]*A[0, 2]*A[2, 3]],
        [ A[3, 1]*A[1, 3]*A[2, 0] + A[3, 3]*A[2, 1]*A[1, 0] + A[1, 1]*A[3, 0]*A[2, 3] - A[1, 0]*A[3, 1]*A[2, 3] - A[3, 0]*A[2, 1]*A[1, 3] - A[1, 1]*A[3, 3]*A[2, 0],
          A[3, 3]*A[0, 1]*A[2, 0] - A[3, 3]*A[0, 0]*A[2, 1] - A[0, 3]*A[3, 1]*A[2, 0] - A[3, 0]*A[0, 1]*A[2, 3] + A[0, 0]*A[3, 1]*A[2, 3] + A[0, 3]*A[3, 0]*A[2, 1],
         -A[0, 0]*A[3, 1]*A[1, 3] + A[0, 3]*A[1, 0]*A[3, 1] - A[3, 3]*A[1, 0]*A[0, 1] + A[1, 1]*A[3, 3]*A[0, 0] - A[1, 1]*A[0, 3]*A[3, 0] + A[3, 0]*A[0, 1]*A[1, 3],
          A[0, 0]*A[2, 1]*A[1, 3] + A[1, 0]*A[0, 1]*A[2, 3] - A[0, 3]*A[2, 1]*A[1, 0] + A[1, 1]*A[0, 3]*A[2, 0] - A[1, 1]*A[0, 0]*A[2, 3] - A[0, 1]*A[1, 3]*A[2, 0]],
        [-A[1, 2]*A[3, 1]*A[2, 0] - A[2, 1]*A[1, 0]*A[3, 2] + A[3, 0]*A[2, 1]*A[1, 2] - A[1, 1]*A[3, 0]*A[2, 2] + A[1, 0]*A[3, 1]*A[2, 2] + A[1, 1]*A[3, 2]*A[2, 0],
         -A[3, 0]*A[2, 1]*A[0, 2] - A[0, 1]*A[3, 2]*A[2, 0] + A[3, 1]*A[0, 2]*A[2, 0] - A[0, 0]*A[3, 1]*A[2, 2] + A[3, 0]*A[0, 1]*A[2, 2] + A[0, 0]*A[2, 1]*A[3, 2],
          A[0, 0]*A[1, 2]*A[3, 1] - A[1, 0]*A[3, 1]*A[0, 2] + A[1, 1]*A[3, 0]*A[0, 2] + A[1, 0]*A[0, 1]*A[3, 2] - A[3, 0]*A[1, 2]*A[0, 1] - A[1, 1]*A[0, 0]*A[3, 2],
         -A[1, 1]*A[0, 2]*A[2, 0] + A[2, 1]*A[1, 0]*A[0, 2] + A[1, 2]*A[0, 1]*A[2, 0] + A[1, 1]*A[0, 0]*A[2, 2] - A[1, 0]*A[0, 1]*A[2, 2] - A[0, 0]*A[2, 1]*A[1, 2]],
        ])


def cofactor_expr(A):
    sh = A.shape()
    ufl_assert(sh[0] == sh[1], "Expecting square matrix.")

    if sh[0] == 2:
        return cofactor_expr_2x2(A)
    elif sh[0] == 3:
        return cofactor_expr_3x3(A)
    elif sh[0] == 4:
        return cofactor_expr_4x4(A)

    error("cofactor_expr not implemented for dimension %s." % sh[0])

def cofactor_expr_2x2(A):
    return as_matrix([[A[1, 1], -A[1, 0]],
                      [-A[0, 1], A[0, 0]]])

def cofactor_expr_3x3(A):
    return as_matrix([
        [A[1, 1]*A[2, 2] - A[2, 1]*A[1, 2], A[2, 0]*A[1, 2] - A[1, 0]*A[2, 2], - A[2, 0]*A[1, 1] + A[1, 0]*A[2, 1]],
        [A[2, 1]*A[0, 2] - A[0, 1]*A[2, 2], A[0, 0]*A[2, 2] - A[2, 0]*A[0, 2], - A[0, 0]*A[2, 1] + A[2, 0]*A[0, 1]],
        [A[0, 1]*A[1, 2] - A[1, 1]*A[0, 2], A[1, 0]*A[0, 2] - A[0, 0]*A[1, 2], - A[1, 0]*A[0, 1] + A[0, 0]*A[1, 1]],
        ])

def cofactor_expr_4x4(A):
    return as_matrix([
        [-A[3, 1]*A[2, 2]*A[1, 3] - A[3, 2]*A[2, 3]*A[1, 1] + A[1, 3]*A[3, 2]*A[2, 1] + A[3, 1]*A[2, 3]*A[1, 2] + A[2, 2]*A[1, 1]*A[3, 3] - A[3, 3]*A[2, 1]*A[1, 2],
         -A[1, 0]*A[2, 2]*A[3, 3] + A[2, 0]*A[3, 3]*A[1, 2] + A[2, 2]*A[1, 3]*A[3, 0] - A[2, 3]*A[3, 0]*A[1, 2] + A[1, 0]*A[3, 2]*A[2, 3] - A[1, 3]*A[3, 2]*A[2, 0],
          A[1, 0]*A[3, 3]*A[2, 1] + A[2, 3]*A[1, 1]*A[3, 0] - A[2, 0]*A[1, 1]*A[3, 3] - A[1, 3]*A[3, 0]*A[2, 1] - A[1, 0]*A[3, 1]*A[2, 3] + A[3, 1]*A[1, 3]*A[2, 0],
          A[3, 0]*A[2, 1]*A[1, 2] + A[1, 0]*A[3, 1]*A[2, 2] + A[3, 2]*A[2, 0]*A[1, 1] - A[2, 2]*A[1, 1]*A[3, 0] - A[3, 1]*A[2, 0]*A[1, 2] - A[1, 0]*A[3, 2]*A[2, 1]],
        [ A[3, 1]*A[2, 2]*A[0, 3] + A[0, 2]*A[3, 3]*A[2, 1] + A[0, 1]*A[3, 2]*A[2, 3] - A[3, 1]*A[0, 2]*A[2, 3] - A[0, 1]*A[2, 2]*A[3, 3] - A[3, 2]*A[0, 3]*A[2, 1],
         -A[2, 2]*A[0, 3]*A[3, 0] - A[0, 2]*A[2, 0]*A[3, 3] - A[3, 2]*A[2, 3]*A[0, 0] + A[2, 2]*A[3, 3]*A[0, 0] + A[0, 2]*A[2, 3]*A[3, 0] + A[3, 2]*A[2, 0]*A[0, 3],
          A[3, 1]*A[2, 3]*A[0, 0] - A[0, 1]*A[2, 3]*A[3, 0] - A[3, 1]*A[2, 0]*A[0, 3] - A[3, 3]*A[0, 0]*A[2, 1] + A[0, 3]*A[3, 0]*A[2, 1] + A[0, 1]*A[2, 0]*A[3, 3],
          A[3, 2]*A[0, 0]*A[2, 1] - A[0, 2]*A[3, 0]*A[2, 1] + A[0, 1]*A[2, 2]*A[3, 0] + A[3, 1]*A[0, 2]*A[2, 0] - A[0, 1]*A[3, 2]*A[2, 0] - A[3, 1]*A[2, 2]*A[0, 0]],
        [ A[3, 1]*A[1, 3]*A[0, 2] - A[0, 2]*A[1, 1]*A[3, 3] - A[3, 1]*A[0, 3]*A[1, 2] + A[3, 2]*A[1, 1]*A[0, 3] + A[0, 1]*A[3, 3]*A[1, 2] - A[0, 1]*A[1, 3]*A[3, 2],
          A[1, 3]*A[3, 2]*A[0, 0] - A[1, 0]*A[3, 2]*A[0, 3] - A[1, 3]*A[0, 2]*A[3, 0] + A[0, 3]*A[3, 0]*A[1, 2] + A[1, 0]*A[0, 2]*A[3, 3] - A[3, 3]*A[0, 0]*A[1, 2],
         -A[1, 0]*A[0, 1]*A[3, 3] + A[0, 1]*A[1, 3]*A[3, 0] - A[3, 1]*A[1, 3]*A[0, 0] - A[1, 1]*A[0, 3]*A[3, 0] + A[1, 0]*A[3, 1]*A[0, 3] + A[1, 1]*A[3, 3]*A[0, 0],
          A[0, 2]*A[1, 1]*A[3, 0] - A[3, 2]*A[1, 1]*A[0, 0] - A[0, 1]*A[3, 0]*A[1, 2] - A[1, 0]*A[3, 1]*A[0, 2] + A[3, 1]*A[0, 0]*A[1, 2] + A[1, 0]*A[0, 1]*A[3, 2]],
        [ A[0, 3]*A[2, 1]*A[1, 2] + A[0, 2]*A[2, 3]*A[1, 1] + A[0, 1]*A[2, 2]*A[1, 3] - A[2, 2]*A[1, 1]*A[0, 3] - A[1, 3]*A[0, 2]*A[2, 1] - A[0, 1]*A[2, 3]*A[1, 2],
          A[1, 0]*A[2, 2]*A[0, 3] + A[1, 3]*A[0, 2]*A[2, 0] - A[1, 0]*A[0, 2]*A[2, 3] - A[2, 0]*A[0, 3]*A[1, 2] - A[2, 2]*A[1, 3]*A[0, 0] + A[2, 3]*A[0, 0]*A[1, 2],
         -A[0, 1]*A[1, 3]*A[2, 0] + A[2, 0]*A[1, 1]*A[0, 3] + A[1, 3]*A[0, 0]*A[2, 1] - A[1, 0]*A[0, 3]*A[2, 1] + A[1, 0]*A[0, 1]*A[2, 3] - A[2, 3]*A[1, 1]*A[0, 0],
          A[1, 0]*A[0, 2]*A[2, 1] - A[0, 2]*A[2, 0]*A[1, 1] + A[0, 1]*A[2, 0]*A[1, 2] + A[2, 2]*A[1, 1]*A[0, 0] - A[1, 0]*A[0, 1]*A[2, 2] - A[0, 0]*A[2, 1]*A[1, 2]]
        ])


def deviatoric_expr(A):
    sh = A.shape()
    ufl_assert(sh[0] == sh[1], "Expecting square matrix.")

    if sh[0] == 2:
        return deviatoric_expr_2x2(A)
    elif sh[0] == 3:
        return deviatoric_expr_3x3(A)

    error("deviatoric_expr not implemented for dimension %s." % sh[0])

def deviatoric_expr_2x2(A):
    return as_matrix([[-1./2*A[1, 1]+1./2*A[0, 0],  A[0, 1]],
                      [A[1, 0],                    1./2*A[1, 1]-1./2*A[0, 0]]])

def deviatoric_expr_3x3(A):
    return as_matrix([[-1./3*A[1, 1]-1./3*A[2, 2]+2./3*A[0, 0],   A[0, 1],   A[0, 2]],
                      [A[1, 0],  2./3*A[1, 1]-1./3*A[2, 2]-1./3*A[0, 0],   A[1, 2]],
                      [A[2, 0],  A[2, 1],   -1./3*A[1, 1]+2./3*A[2, 2]-1./3*A[0, 0]]])
