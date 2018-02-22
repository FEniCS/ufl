# -*- coding: utf-8 -*-
"""Compound tensor algebra operations."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
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

from ufl.log import error
from ufl.utils.str import as_native_strings
from ufl.core.expr import ufl_err_str
from ufl.core.ufl_type import ufl_type
from ufl.constantvalue import Zero
from ufl.algebra import Operator, Conj
from ufl.precedence import parstr
from ufl.sorting import sorted_expr
from ufl.index_combination_utils import merge_nonoverlapping_indices

# Algebraic operations on tensors:
# FloatValues:
#   dot(a,b)      = a*b
#   inner(a,b)    = a*b
#   outer(a,b)    = a*b
# Vectors:
#   dot(u,v)      = u_i v_i
#   inner(u,v)    = u_i v_i
#   outer(u,v)    = A | A_ij = u_i v_j
# Matrices:
#   dot(A,B)      = C | C_ij = A_{ik} B_{kj}
#   inner(A,B)    = A_{ij} B_{ij}
#   outer(A,B)    = C | C_ijkl = A_ij B_kl
# Combined:
#   dot(A,u)      = v | v_i = A_{ik} u_k
#   inner(A,u)    = v | v_i = A_{ik} u_k
#   outer(A,u)    = C | C_ijk = B_ij u_k
#   dot(u,B)      = v | v_i = u_k B_{ki}
#   inner(u,B)    = v | v_i = u_k B_{ki}
#   outer(u,B)    = C | C_ijk = u_i B_jk
#
# Argument requirements:
#   dot(x,y):   last index of x has same dimension as first index of y
#   inner(x,y): shape of x equals the shape of y


# --- Classes representing compound tensor algebra operations ---

@ufl_type(is_abstract=True)
class CompoundTensorOperator(Operator):
    __slots__ = ()

    def __init__(self, operands):
        Operator.__init__(self, operands)

# TODO: Use this and make Sum handle scalars only?
#       This would simplify some algorithms. The only
#       problem is we can't use + in many algorithms because
#       this type should be expanded by expand_compounds.
# class TensorSum(CompoundTensorOperator):
#     "Sum of nonscalar expressions."
#     pass

# TODO: Use this similarly to TensorSum?
#       This would simplify some algorithms. The only
#       problem is we can't use / in many algorithms because
#       this type should be expanded by expand_compounds.
# class TensorDivision(CompoundTensorOperator):
#     "Division of nonscalar expression with a scalar expression."
#     pass

# TODO: Use this similarly to TensorSum?
#       This would simplify some algorithms. The only
#       problem is we can't use * in many algorithms because
#       this type should be expanded by expand_compounds.
# class MatrixProduct(CompoundTensorOperator):
#     "Product of a matrix with a matrix or vector."
#     pass

# TODO: Use this similarly to TensorSum?
#       This would simplify some algorithms. The only
#       problem is we can't use abs in many algorithms because
#       this type should be expanded by expand_compounds.
# class TensorAbs(CompoundTensorOperator):
#     "Absolute value of nonscalar expression."
#     pass


@ufl_type(is_shaping=True, num_ops=1, inherit_indices_from_operand=0)
class Transposed(CompoundTensorOperator):
    __slots__ = ()

    def __new__(cls, A):
        if isinstance(A, Zero):
            a, b = A.ufl_shape
            return Zero((b, a), A.ufl_free_indices, A.ufl_index_dimensions)
        return CompoundTensorOperator.__new__(cls)

    def __init__(self, A):
        CompoundTensorOperator.__init__(self, (A,))
        if len(A.ufl_shape) != 2:
            error("Transposed is only defined for rank 2 tensors.")

    @property
    def ufl_shape(self):
        s = self.ufl_operands[0].ufl_shape
        return (s[1], s[0])

    def __str__(self):
        return "%s^T" % parstr(self.ufl_operands[0], self)


@ufl_type(num_ops=2)
class Outer(CompoundTensorOperator):
    __slots__ = as_native_strings(("ufl_free_indices",
                                   "ufl_index_dimensions"))

    def __new__(cls, a, b):
        ash, bsh = a.ufl_shape, b.ufl_shape
        if isinstance(a, Zero) or isinstance(b, Zero):
            fi, fid = merge_nonoverlapping_indices(a, b)
            return Zero(ash + bsh, fi, fid)
        if ash == () or bsh == ():
            return Conj(a) * b
        return CompoundTensorOperator.__new__(cls)

    def __init__(self, a, b):
        CompoundTensorOperator.__init__(self, (a, b))
        fi, fid = merge_nonoverlapping_indices(a, b)
        self.ufl_free_indices = fi
        self.ufl_index_dimensions = fid

    @property
    def ufl_shape(self):
        return self.ufl_operands[0].ufl_shape + self.ufl_operands[1].ufl_shape

    def __str__(self):
        return "%s (X) %s" % (parstr(self.ufl_operands[0], self),
                              parstr(self.ufl_operands[1], self))


@ufl_type(num_ops=2)
class Inner(CompoundTensorOperator):
    __slots__ = as_native_strings(("ufl_free_indices",
                                   "ufl_index_dimensions"))

    def __new__(cls, a, b):
        # Checks
        ash, bsh = a.ufl_shape, b.ufl_shape
        if ash != bsh:
            error("Shapes do not match: %s and %s." % (ufl_err_str(a), ufl_err_str(b)))

        # Simplification
        if isinstance(a, Zero) or isinstance(b, Zero):
            fi, fid = merge_nonoverlapping_indices(a, b)
            return Zero((), fi, fid)
        elif ash == ():
            return a*Conj(b)
        # sort operands for unique representation,
        # must be independent of various counts etc.
        # as explained in cmp_expr
        if (a, b) != tuple(sorted_expr((a, b))):
            return Conj(Inner(b, a))

        return CompoundTensorOperator.__new__(cls)

    def __init__(self, a, b):

        CompoundTensorOperator.__init__(self, (a, b))

        fi, fid = merge_nonoverlapping_indices(a, b)
        self.ufl_free_indices = fi
        self.ufl_index_dimensions = fid

    ufl_shape = ()

    def __str__(self):
        return "%s : %s" % (parstr(self.ufl_operands[0], self),
                            parstr(self.ufl_operands[1], self))


@ufl_type(num_ops=2)
class Dot(CompoundTensorOperator):
    __slots__ = as_native_strings(("ufl_free_indices",
                                   "ufl_index_dimensions"))

    def __new__(cls, a, b):
        ash = a.ufl_shape
        bsh = b.ufl_shape
        ar, br = len(ash), len(bsh)
        scalar = (ar == 0 and br == 0)

        # Checks
        if not ((ar >= 1 and br >= 1) or scalar):
            error("Dot product requires non-scalar arguments, "
                  "got arguments with ranks %d and %d." % (ar, br))
        if not (scalar or ash[-1] == bsh[0]):
            error("Dimension mismatch in dot product.")

        # Simplification
        if isinstance(a, Zero) or isinstance(b, Zero):
            shape = ash[:-1] + bsh[1:]
            fi, fid = merge_nonoverlapping_indices(a, b)
            return Zero(shape, fi, fid)
        elif scalar:  # TODO: Move this to def dot()?
            return a * Conj(b)

        return CompoundTensorOperator.__new__(cls)

    def __init__(self, a, b):
        CompoundTensorOperator.__init__(self, (a, b))
        fi, fid = merge_nonoverlapping_indices(a, b)
        self.ufl_free_indices = fi
        self.ufl_index_dimensions = fid

    @property
    def ufl_shape(self):
        return self.ufl_operands[0].ufl_shape[:-1] + self.ufl_operands[1].ufl_shape[1:]

    def __str__(self):
        return "%s . %s" % (parstr(self.ufl_operands[0], self),
                            parstr(self.ufl_operands[1], self))


@ufl_type(num_ops=2)
class Cross(CompoundTensorOperator):
    __slots__ = as_native_strings(("ufl_free_indices",
                                   "ufl_index_dimensions"))

    def __new__(cls, a, b):
        ash = a.ufl_shape
        bsh = b.ufl_shape

        # Checks
        if not (len(ash) == 1 and ash == bsh):
            error("Cross product requires arguments of rank 1, got %s and %s." % (
                ufl_err_str(a), ufl_err_str(b)))

        # Simplification
        if isinstance(a, Zero) or isinstance(b, Zero):
            fi, fid = merge_nonoverlapping_indices(a, b)
            return Zero(ash, fi, fid)

        return CompoundTensorOperator.__new__(cls)

    def __init__(self, a, b):
        CompoundTensorOperator.__init__(self, (a, b))
        fi, fid = merge_nonoverlapping_indices(a, b)
        self.ufl_free_indices = fi
        self.ufl_index_dimensions = fid

    ufl_shape = (3,)

    def __str__(self):
        return "%s x %s" % (parstr(self.ufl_operands[0], self),
                            parstr(self.ufl_operands[1], self))


@ufl_type(num_ops=1, inherit_indices_from_operand=0)
class Trace(CompoundTensorOperator):
    __slots__ = ()

    def __new__(cls, A):
        # Checks
        if len(A.ufl_shape) != 2:
            error("Trace of tensor with rank != 2 is undefined.")

        # Simplification
        if isinstance(A, Zero):
            return Zero((), A.ufl_free_indices, A.ufl_index_dimensions)

        return CompoundTensorOperator.__new__(cls)

    def __init__(self, A):
        CompoundTensorOperator.__init__(self, (A,))

    ufl_shape = ()

    def __str__(self):
        return "tr(%s)" % self.ufl_operands[0]


@ufl_type(is_scalar=True, num_ops=1)
class Determinant(CompoundTensorOperator):
    __slots__ = ()

    def __new__(cls, A):
        sh = A.ufl_shape
        r = len(sh)
        Afi = A.ufl_free_indices

        # Checks
        if r not in (0, 2):
            error("Determinant of tensor with rank != 2 is undefined.")
        if r == 2 and sh[0] != sh[1]:
            error("Cannot take determinant of rectangular rank 2 tensor.")
        if Afi:
            error("Not expecting free indices in determinant.")

        # Simplification
        if isinstance(A, Zero):
            return Zero((), Afi, A.ufl_index_dimensions)
        if r == 0:
            return A

        return CompoundTensorOperator.__new__(cls)

    def __init__(self, A):
        CompoundTensorOperator.__init__(self, (A,))

    def __str__(self):
        return "det(%s)" % self.ufl_operands[0]


# TODO: Drop Inverse and represent it as product of Determinant and
# Cofactor?
@ufl_type(is_index_free=True, num_ops=1)
class Inverse(CompoundTensorOperator):
    __slots__ = ()

    def __new__(cls, A):
        sh = A.ufl_shape
        r = len(sh)

        # Checks
        if A.ufl_free_indices:
            error("Not expecting free indices in Inverse.")
        if isinstance(A, Zero):
            error("Division by zero!")

        # Simplification
        if r == 0:
            return 1 / A

        # More checks
        if r != 2:
            error("Inverse of tensor with rank != 2 is undefined.")
        if sh[0] != sh[1]:
            error("Cannot take inverse of rectangular matrix with dimensions %s." % (sh,))

        return CompoundTensorOperator.__new__(cls)

    def __init__(self, A):
        CompoundTensorOperator.__init__(self, (A,))

    @property
    def ufl_shape(self):
        return self.ufl_operands[0].ufl_shape

    def __str__(self):
        return "%s^-1" % parstr(self.ufl_operands[0], self)


@ufl_type(is_index_free=True, num_ops=1)
class Cofactor(CompoundTensorOperator):
    __slots__ = ()

    def __init__(self, A):
        CompoundTensorOperator.__init__(self, (A,))

        # Checks
        sh = A.ufl_shape
        if len(sh) != 2:
            error("Cofactor of tensor with rank != 2 is undefined.")
        if sh[0] != sh[1]:
            error("Cannot take cofactor of rectangular matrix with dimensions %s." % (sh,))
        if A.ufl_free_indices:
            error("Not expecting free indices in Cofactor.")
        if isinstance(A, Zero):
            error("Cannot take cofactor of zero matrix.")

    @property
    def ufl_shape(self):
        return self.ufl_operands[0].ufl_shape

    def __str__(self):
        return "cofactor(%s)" % self.ufl_operands[0]


@ufl_type(num_ops=1, inherit_shape_from_operand=0, inherit_indices_from_operand=0)
class Deviatoric(CompoundTensorOperator):
    __slots__ = ()

    def __new__(cls, A):
        sh = A.ufl_shape

        # Checks
        if len(sh) != 2:
            error("Deviatoric part of tensor with rank != 2 is undefined.")
        if sh[0] != sh[1]:
            error("Cannot take deviatoric part of rectangular matrix with dimensions %s." % (sh,))
        if A.ufl_free_indices:
            error("Not expecting free indices in Deviatoric.")

        # Simplification
        if isinstance(A, Zero):
            return Zero(A.ufl_shape, A.ufl_free_indices, A.ufl_index_dimensions)

        return CompoundTensorOperator.__new__(cls)

    def __init__(self, A):
        CompoundTensorOperator.__init__(self, (A,))

    def __str__(self):
        return "dev(%s)" % self.ufl_operands[0]


@ufl_type(num_ops=1, inherit_shape_from_operand=0, inherit_indices_from_operand=0)
class Skew(CompoundTensorOperator):
    __slots__ = ()

    def __new__(cls, A):
        sh = A.ufl_shape
        Afi = A.ufl_free_indices

        # Checks
        if len(sh) != 2:
            error("Skew symmetric part of tensor with rank != 2 is undefined.")
        if sh[0] != sh[1]:
            error("Cannot take skew part of rectangular matrix with dimensions %s." % (sh,))
        if Afi:
            error("Not expecting free indices in Skew.")

        # Simplification
        if isinstance(A, Zero):
            return Zero(A.ufl_shape, Afi, A.ufl_index_dimensions)

        return CompoundTensorOperator.__new__(cls)

    def __init__(self, A):
        CompoundTensorOperator.__init__(self, (A,))

    def __str__(self):
        return "skew(%s)" % self.ufl_operands[0]


@ufl_type(num_ops=1, inherit_shape_from_operand=0, inherit_indices_from_operand=0)
class Sym(CompoundTensorOperator):
    __slots__ = ()

    def __new__(cls, A):
        sh = A.ufl_shape
        Afi = A.ufl_free_indices

        # Checks
        if len(sh) != 2:
            error("Symmetric part of tensor with rank != 2 is undefined.")
        if sh[0] != sh[1]:
            error("Cannot take symmetric part of rectangular matrix with dimensions %s." % (sh,))
        if Afi:
            error("Not expecting free indices in Sym.")

        # Simplification
        if isinstance(A, Zero):
            return Zero(A.ufl_shape, Afi, A.ufl_index_dimensions)

        return CompoundTensorOperator.__new__(cls)

    def __init__(self, A):
        CompoundTensorOperator.__init__(self, (A,))

    def __str__(self):
        return "sym(%s)" % self.ufl_operands[0]
