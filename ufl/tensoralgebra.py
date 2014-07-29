"""Compound tensor algebra operations."""

# Copyright (C) 2008-2014 Martin Sandve Alnes
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

from ufl.log import warning
from ufl.assertions import ufl_assert
from ufl.constantvalue import Zero
from ufl.algebra import AlgebraOperator
from ufl.precedence import parstr
from ufl.sorting import sorted_expr
from ufl.core.ufl_type import ufl_type

def merge_indices(a, b):
    ai = a.free_indices()
    bi = b.free_indices()
    #free_indices = unique_indices(ai + bi) # if repeated indices are allowed, do this instead of the next three lines
    ri = set(ai) & set(bi)
    ufl_assert(not ri, "Not expecting repeated indices.")
    free_indices = ai+bi

    aid = a.index_dimensions()
    bid = b.index_dimensions()
    index_dimensions = dict(aid)
    index_dimensions.update(bid)

    return free_indices, index_dimensions

### Algebraic operations on tensors:
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
class CompoundTensorOperator(AlgebraOperator):
    __slots__ = ()
    def __init__(self, operands):
        AlgebraOperator.__init__(self, operands)

# TODO: Use this and make Sum handle scalars only?
#       This would simplify some algorithms. The only
#       problem is we can't use + in many algorithms because
#       this type should be expanded by expand_compounds.
#class TensorSum(CompoundTensorOperator):
#    "Sum of nonscalar expressions."
#    pass

# TODO: Use this similarly to TensorSum?
#       This would simplify some algorithms. The only
#       problem is we can't use / in many algorithms because
#       this type should be expanded by expand_compounds.
#class TensorDivision(CompoundTensorOperator):
#    "Division of nonscalar expression with a scalar expression."
#    pass

# TODO: Use this similarly to TensorSum?
#       This would simplify some algorithms. The only
#       problem is we can't use * in many algorithms because
#       this type should be expanded by expand_compounds.
#class MatrixProduct(CompoundTensorOperator):
#    "Product of a matrix with a matrix or vector."
#    pass

# TODO: Use this similarly to TensorSum?
#       This would simplify some algorithms. The only
#       problem is we can't use abs in many algorithms because
#       this type should be expanded by expand_compounds.
#class TensorAbs(CompoundTensorOperator):
#    "Absolute value of nonscalar expression."
#    pass

@ufl_type(is_shaping=True, num_ops=1, inherit_indices_from_operand=0)
class Transposed(CompoundTensorOperator):
    __slots__ = ()

    def __new__(cls, A):
        if isinstance(A, Zero):
            a, b = A.ufl_shape
            return Zero((b, a), A.free_indices(), A.index_dimensions())
        return CompoundTensorOperator.__new__(cls)

    def __init__(self, A):
        ufl_assert(A.rank() == 2, "Transposed is only defined for rank 2 tensors.")

        CompoundTensorOperator.__init__(self, (A,))

    @property
    def ufl_shape(self):
        s = self.ufl_operands[0].ufl_shape
        return (s[1], s[0])

    def __str__(self):
        return "%s^T" % parstr(self.ufl_operands[0], self)

    def __repr__(self):
        return "Transposed(%r)" % self.ufl_operands[0]

@ufl_type(num_ops=2)
class Outer(CompoundTensorOperator):
    __slots__ = ("_free_indices", "_index_dimensions")

    def __new__(cls, a, b):
        ash, bsh = a.ufl_shape, b.ufl_shape
        if isinstance(a, Zero) or isinstance(b, Zero):
            free_indices, index_dimensions = merge_indices(a, b)
            return Zero(ash + bsh, free_indices, index_dimensions)
        if ash == () or bsh == ():
            return a * b
        return CompoundTensorOperator.__new__(cls)

    def __init__(self, a, b):
        CompoundTensorOperator.__init__(self, (a, b))
        self._free_indices, self._index_dimensions = merge_indices(a, b)

    def free_indices(self):
        return self._free_indices

    def index_dimensions(self):
        return self._index_dimensions

    @property
    def ufl_shape(self):
        return self.ufl_operands[0].ufl_shape + self.ufl_operands[1].ufl_shape

    def __str__(self):
        return "%s (X) %s" % (parstr(self.ufl_operands[0], self), parstr(self.ufl_operands[1], self))

    def __repr__(self):
        return "Outer(%r, %r)" % (self.ufl_operands[0], self.ufl_operands[1])

@ufl_type(num_ops=2)
class Inner(CompoundTensorOperator):
    __slots__ = ("_free_indices", "_index_dimensions")

    def __new__(cls, a, b):
        ash, bsh = a.ufl_shape, b.ufl_shape
        ufl_assert(ash == bsh, "Shape mismatch.")
        if isinstance(a, Zero) or isinstance(b, Zero):
            free_indices, index_dimensions = merge_indices(a, b)
            return Zero((), free_indices, index_dimensions)
        if ash == ():
            return a*b
        return CompoundTensorOperator.__new__(cls)

    def __init__(self, a, b):
        # sort operands for unique representation,
        # must be independent of various counts etc.
        # as explained in cmp_expr
        a, b = sorted_expr((a, b))

        # old version, slow and unsafe:
        #a, b = sorted((a,b), key = lambda x: repr(x))

        CompoundTensorOperator.__init__(self, (a, b))

        self._free_indices, self._index_dimensions = merge_indices(a, b)

    ufl_shape = ()

    def free_indices(self):
        return self._free_indices

    def index_dimensions(self):
        return self._index_dimensions

    def __str__(self):
        return "%s : %s" % (parstr(self.ufl_operands[0], self), parstr(self.ufl_operands[1], self))

    def __repr__(self):
        return "Inner(%r, %r)" % (self.ufl_operands[0], self.ufl_operands[1])

@ufl_type(num_ops=2)
class Dot(CompoundTensorOperator):
    __slots__ = ("_free_indices", "_index_dimensions")

    def __new__(cls, a, b):
        ash = a.ufl_shape
        bsh = b.ufl_shape
        ar, br = len(ash), len(bsh)
        scalar = (ar == 0 and br == 0)
        ufl_assert((ar >= 1 and br >= 1) or scalar,
            "Dot product requires non-scalar arguments, "\
            "got arguments with ranks %d and %d." % \
            (ar, br))
        ufl_assert(scalar or ash[-1] == bsh[0], "Dimension mismatch in dot product.")
        if isinstance(a, Zero) or isinstance(b, Zero):
            shape = ash[:-1] + bsh[1:]
            free_indices, index_dimensions = merge_indices(a, b)
            return Zero(shape, free_indices, index_dimensions)
        if scalar: # TODO: Move this to def dot()?
            return a * b
        return CompoundTensorOperator.__new__(cls)

    def __init__(self, a, b):
        CompoundTensorOperator.__init__(self, (a, b))
        self._free_indices, self._index_dimensions = merge_indices(a, b)

    @property
    def ufl_shape(self):
        return self.ufl_operands[0].ufl_shape[:-1] + self.ufl_operands[1].ufl_shape[1:]

    def free_indices(self):
        return self._free_indices

    def index_dimensions(self):
        return self._index_dimensions

    def __str__(self):
        return "%s . %s" % (parstr(self.ufl_operands[0], self), parstr(self.ufl_operands[1], self))

    def __repr__(self):
        return "Dot(%r, %r)" % (self.ufl_operands[0], self.ufl_operands[1])

@ufl_type(num_ops=2)
class Cross(CompoundTensorOperator):
    __slots__ = ("_free_indices", "_index_dimensions")

    def __new__(cls, a, b):
        ash, bsh = a.ufl_shape, b.ufl_shape
        ufl_assert(len(ash) == 1 and ash == bsh,
            "Cross product requires arguments of rank 1.")

        if isinstance(a, Zero) or isinstance(b, Zero):
            free_indices, index_dimensions = merge_indices(a, b)
            return Zero(ash, free_indices, index_dimensions)

        return CompoundTensorOperator.__new__(cls)

    def __init__(self, a, b):
        CompoundTensorOperator.__init__(self, (a, b))
        self._free_indices, self._index_dimensions = merge_indices(a, b)

    ufl_shape = (3,)

    def free_indices(self):
        return self._free_indices

    def index_dimensions(self):
        return self._index_dimensions

    def __str__(self):
        return "%s x %s" % (parstr(self.ufl_operands[0], self), parstr(self.ufl_operands[1], self))

    def __repr__(self):
        return "Cross(%r, %r)" % (self.ufl_operands[0], self.ufl_operands[1])

@ufl_type(num_ops=1, inherit_indices_from_operand=0)
class Trace(CompoundTensorOperator):
    __slots__ = ()

    def __new__(cls, A):
        ufl_assert(A.rank() == 2, "Trace of tensor with rank != 2 is undefined.")
        if isinstance(A, Zero):
            return Zero((), A.free_indices(), A.index_dimensions())
        return CompoundTensorOperator.__new__(cls)

    def __init__(self, A):
        CompoundTensorOperator.__init__(self, (A,))

    ufl_shape = ()

    def __str__(self):
        return "tr(%s)" % self.ufl_operands[0]

    def __repr__(self):
        return "Trace(%r)" % self.ufl_operands[0]

@ufl_type(is_scalar=True, num_ops=1)
class Determinant(CompoundTensorOperator):
    __slots__ = ()

    def __new__(cls, A):
        sh = A.ufl_shape
        r = len(sh)
        ufl_assert(r == 0 or r == 2,
            "Determinant of tensor with rank != 2 is undefined.")
        ufl_assert(r == 0 or sh[0] == sh[1],
            "Cannot take determinant of rectangular rank 2 tensor.")
        ufl_assert(not A.free_indices(),
            "Not expecting free indices in determinant.")
        if isinstance(A, Zero):
            return Zero((), A.free_indices(), A.index_dimensions())
        if r == 0:
            return A
        return CompoundTensorOperator.__new__(cls)

    def __init__(self, A):
        CompoundTensorOperator.__init__(self, (A,))

    def __str__(self):
        return "det(%s)" % self.ufl_operands[0]

    def __repr__(self):
        return "Determinant(%r)" % self.ufl_operands[0]

# TODO: Drop Inverse and represent it as product of Determinant and Cofactor?
@ufl_type(is_index_free=True, num_ops=1)
class Inverse(CompoundTensorOperator):
    __slots__ = ()

    def __new__(cls, A):
        sh = A.ufl_shape
        r = len(sh)
        if A.free_indices():
            error("Not expecting free indices in Inverse.")
        if isinstance(A, Zero):
            error("Division by zero!")

        if r == 0:
            return 1 / A

        if r != 2:
            error("Inverse of tensor with rank != 2 is undefined.")
        if sh[0] != sh[1]:
            error("Cannot take inverse of rectangular matrix with dimensions %s." % repr(sh))
        return CompoundTensorOperator.__new__(cls)

    def __init__(self, A):
        CompoundTensorOperator.__init__(self, (A,))

    @property
    def ufl_shape(self):
        return self.ufl_operands[0].ufl_shape

    def __str__(self):
        return "%s^-1" % parstr(self.ufl_operands[0], self)

    def __repr__(self):
        return "Inverse(%r)" % self.ufl_operands[0]

@ufl_type(is_index_free=True, num_ops=1)
class Cofactor(CompoundTensorOperator):
    __slots__ = ()

    def __init__(self, A):
        CompoundTensorOperator.__init__(self, (A,))

        sh = A.ufl_shape
        ufl_assert(len(sh) == 2, "Cofactor of tensor with rank != 2 is undefined.")
        if sh[0] != sh[1]:
            error("Cannot take cofactor of rectangular matrix with dimensions %s." % repr(sh))
        ufl_assert(not A.free_indices(), "Not expecting free indices in Cofactor.")
        ufl_assert(not isinstance(A, Zero), "Cannot take cofactor of zero matrix.")

    @property
    def ufl_shape(self):
        return self.ufl_operands[0].ufl_shape

    def __str__(self):
        return "cofactor(%s)" % self.ufl_operands[0]

    def __repr__(self):
        return "Cofactor(%r)" % self.ufl_operands[0]

@ufl_type(num_ops=1, inherit_shape_from_operand=0, inherit_indices_from_operand=0)
class Deviatoric(CompoundTensorOperator):
    __slots__ = ()

    def __new__(cls, A):
        sh = A.ufl_shape
        ufl_assert(len(sh) == 2, "Deviatoric part of tensor with rank != 2 is undefined.")
        if sh[0] != sh[1]:
            error("Cannot take deviatoric part of rectangular matrix with dimensions %s." % repr(sh))
        ufl_assert(not A.free_indices(), "Not expecting free indices in Deviatoric.")
        if isinstance(A, Zero):
            return Zero(A.ufl_shape, A.free_indices(), A.index_dimensions())
        return CompoundTensorOperator.__new__(cls)

    def __init__(self, A):
        CompoundTensorOperator.__init__(self, (A,))

    def __str__(self):
        return "dev(%s)" % self.ufl_operands[0]

    def __repr__(self):
        return "Deviatoric(%r)" % self.ufl_operands[0]

@ufl_type(num_ops=1, inherit_shape_from_operand=0, inherit_indices_from_operand=0)
class Skew(CompoundTensorOperator):
    __slots__ = ()

    def __new__(cls, A):
        sh = A.ufl_shape
        ufl_assert(len(sh) == 2, "Skew symmetric part of tensor with rank != 2 is undefined.")
        if sh[0] != sh[1]:
            error("Cannot take skew part of rectangular matrix with dimensions %s." % repr(sh))
        ufl_assert(not A.free_indices(), "Not expecting free indices in Skew.")
        if isinstance(A, Zero):
            return Zero(A.ufl_shape, A.free_indices(), A.index_dimensions())
        return CompoundTensorOperator.__new__(cls)

    def __init__(self, A):
        CompoundTensorOperator.__init__(self, (A,))

    def __str__(self):
        return "skew(%s)" % self.ufl_operands[0]

    def __repr__(self):
        return "Skew(%r)" % self.ufl_operands[0]

@ufl_type(num_ops=1, inherit_shape_from_operand=0, inherit_indices_from_operand=0)
class Sym(CompoundTensorOperator):
    __slots__ = ()

    def __new__(cls, A):
        sh = A.ufl_shape
        ufl_assert(len(sh) == 2, "Symmetric part of tensor with rank != 2 is undefined.")
        if sh[0] != sh[1]:
            error("Cannot take symmetric part of rectangular matrix with dimensions %s." % repr(sh))
        ufl_assert(not A.free_indices(), "Not expecting free indices in Sym.")
        if isinstance(A, Zero):
            return Zero(A.ufl_shape, A.free_indices(), A.index_dimensions())
        return CompoundTensorOperator.__new__(cls)

    def __init__(self, A):
        CompoundTensorOperator.__init__(self, (A,))

    def __str__(self):
        return "sym(%s)" % self.ufl_operands[0]

    def __repr__(self):
        return "Sym(%r)" % self.ufl_operands[0]
