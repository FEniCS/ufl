"""Compound tensor algebra operations."""
# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.algebra import Conj, Operator
from ufl.constantvalue import Zero
from ufl.core.expr import ufl_err_str
from ufl.core.ufl_type import ufl_type
from ufl.index_combination_utils import merge_nonoverlapping_indices
from ufl.precedence import parstr
from ufl.sorting import sorted_expr

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
    """Compount tensor operator."""

    __slots__ = ()

    def __init__(self, operands):
        """Initialise."""
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
    """Transposed tensor."""

    __slots__ = ()

    def __new__(cls, A):
        """Create new Transposed."""
        if isinstance(A, Zero):
            a, b = A.ufl_shape
            return Zero((b, a), A.ufl_free_indices, A.ufl_index_dimensions)
        return CompoundTensorOperator.__new__(cls)

    def __init__(self, A):
        """Initialise."""
        CompoundTensorOperator.__init__(self, (A,))
        if len(A.ufl_shape) != 2:
            raise ValueError("Transposed is only defined for rank 2 tensors.")

    @property
    def ufl_shape(self):
        """Get the UFL shape."""
        s = self.ufl_operands[0].ufl_shape
        return (s[1], s[0])

    def __str__(self):
        """Format as a string."""
        return "%s^T" % parstr(self.ufl_operands[0], self)


@ufl_type(num_ops=2)
class Outer(CompoundTensorOperator):
    """Outer."""

    __slots__ = ("ufl_free_indices", "ufl_index_dimensions")

    def __new__(cls, a, b):
        """Create new Outer."""
        ash, bsh = a.ufl_shape, b.ufl_shape
        if isinstance(a, Zero) or isinstance(b, Zero):
            fi, fid = merge_nonoverlapping_indices(a, b)
            return Zero(ash + bsh, fi, fid)
        if ash == () or bsh == ():
            return Conj(a) * b
        return CompoundTensorOperator.__new__(cls)

    def __init__(self, a, b):
        """Initialise."""
        CompoundTensorOperator.__init__(self, (a, b))
        fi, fid = merge_nonoverlapping_indices(a, b)
        self.ufl_free_indices = fi
        self.ufl_index_dimensions = fid

    @property
    def ufl_shape(self):
        """Get the UFL shape."""
        return self.ufl_operands[0].ufl_shape + self.ufl_operands[1].ufl_shape

    def __str__(self):
        """Format as a string."""
        return "%s (X) %s" % (parstr(self.ufl_operands[0], self),
                              parstr(self.ufl_operands[1], self))


@ufl_type(num_ops=2)
class Inner(CompoundTensorOperator):
    """Inner."""

    __slots__ = ("ufl_free_indices", "ufl_index_dimensions")

    def __new__(cls, a, b):
        """Create new Inner."""
        # Checks
        ash, bsh = a.ufl_shape, b.ufl_shape
        if ash != bsh:
            raise ValueError(f"Shapes do not match: {ufl_err_str(a)} and {ufl_err_str(b)}")

        # Simplification
        if isinstance(a, Zero) or isinstance(b, Zero):
            fi, fid = merge_nonoverlapping_indices(a, b)
            return Zero((), fi, fid)
        elif ash == ():
            return a * Conj(b)
        # sort operands for unique representation,
        # must be independent of various counts etc.
        # as explained in cmp_expr
        if (a, b) != tuple(sorted_expr((a, b))):
            return Conj(Inner(b, a))

        return CompoundTensorOperator.__new__(cls)

    def __init__(self, a, b):
        """Initialise."""
        CompoundTensorOperator.__init__(self, (a, b))

        fi, fid = merge_nonoverlapping_indices(a, b)
        self.ufl_free_indices = fi
        self.ufl_index_dimensions = fid

    ufl_shape = ()

    def __str__(self):
        """Format as a string."""
        return "%s : %s" % (parstr(self.ufl_operands[0], self),
                            parstr(self.ufl_operands[1], self))


@ufl_type(num_ops=2)
class Dot(CompoundTensorOperator):
    """Dot."""

    __slots__ = ("ufl_free_indices", "ufl_index_dimensions")

    def __new__(cls, a, b):
        """Create new Dot."""
        ash = a.ufl_shape
        bsh = b.ufl_shape
        ar, br = len(ash), len(bsh)
        scalar = (ar == 0 and br == 0)

        # Checks
        if not ((ar >= 1 and br >= 1) or scalar):
            raise ValueError(
                "Dot product requires non-scalar arguments, "
                f"got arguments with ranks {ar} and {br}.")
        if not (scalar or ash[-1] == bsh[0]):
            raise ValueError("Dimension mismatch in dot product.")

        # Simplification
        if isinstance(a, Zero) or isinstance(b, Zero):
            shape = ash[:-1] + bsh[1:]
            fi, fid = merge_nonoverlapping_indices(a, b)
            return Zero(shape, fi, fid)
        elif scalar:  # TODO: Move this to def dot()?
            return a * b

        return CompoundTensorOperator.__new__(cls)

    def __init__(self, a, b):
        """Initialise."""
        CompoundTensorOperator.__init__(self, (a, b))
        fi, fid = merge_nonoverlapping_indices(a, b)
        self.ufl_free_indices = fi
        self.ufl_index_dimensions = fid

    @property
    def ufl_shape(self):
        """Get the UFL shape."""
        return self.ufl_operands[0].ufl_shape[:-1] + self.ufl_operands[1].ufl_shape[1:]

    def __str__(self):
        """Format as a string."""
        return "%s . %s" % (parstr(self.ufl_operands[0], self),
                            parstr(self.ufl_operands[1], self))


@ufl_type(is_index_free=True, num_ops=1)
class Perp(CompoundTensorOperator):
    """Perp."""

    __slots__ = ()

    def __new__(cls, A):
        """Create new Perp."""
        sh = A.ufl_shape

        # Checks
        if not len(sh) == 1:
            raise ValueError(f"Perp requires arguments of rank 1, got {ufl_err_str(A)}")
        if not sh[0] == 2:
            raise ValueError(f"Perp can only work on 2D vectors, got {ufl_err_str(A)}")

        # Simplification
        if isinstance(A, Zero):
            return Zero(sh, A.ufl_free_indices, A.ufl_index_dimensions)

        return CompoundTensorOperator.__new__(cls)

    def __init__(self, A):
        """Initialise."""
        CompoundTensorOperator.__init__(self, (A,))

    ufl_shape = (2,)

    def __str__(self):
        """Format as a string."""
        return "perp(%s)" % self.ufl_operands[0]


@ufl_type(num_ops=2)
class Cross(CompoundTensorOperator):
    """Cross."""

    __slots__ = ("ufl_free_indices", "ufl_index_dimensions")

    def __new__(cls, a, b):
        """Create new Cross."""
        ash = a.ufl_shape
        bsh = b.ufl_shape

        # Checks
        if not (len(ash) == 1 and ash == bsh):
            raise ValueError(
                f"Cross product requires arguments of rank 1, got {ufl_err_str(a)} "
                f"and {ufl_err_str(b)}.")

        # Simplification
        if isinstance(a, Zero) or isinstance(b, Zero):
            fi, fid = merge_nonoverlapping_indices(a, b)
            return Zero(ash, fi, fid)

        return CompoundTensorOperator.__new__(cls)

    def __init__(self, a, b):
        """Initialise."""
        CompoundTensorOperator.__init__(self, (a, b))
        fi, fid = merge_nonoverlapping_indices(a, b)
        self.ufl_free_indices = fi
        self.ufl_index_dimensions = fid

    ufl_shape = (3,)

    def __str__(self):
        """Format as a string."""
        return "%s x %s" % (parstr(self.ufl_operands[0], self),
                            parstr(self.ufl_operands[1], self))


@ufl_type(num_ops=1, inherit_indices_from_operand=0)
class Trace(CompoundTensorOperator):
    """Trace."""

    __slots__ = ()

    def __new__(cls, A):
        """Create new Trace."""
        # Checks
        if len(A.ufl_shape) != 2:
            raise ValueError("Trace of tensor with rank != 2 is undefined.")

        # Simplification
        if isinstance(A, Zero):
            return Zero((), A.ufl_free_indices, A.ufl_index_dimensions)

        return CompoundTensorOperator.__new__(cls)

    def __init__(self, A):
        """Initialise."""
        CompoundTensorOperator.__init__(self, (A,))

    ufl_shape = ()

    def __str__(self):
        """Format as a string."""
        return "tr(%s)" % self.ufl_operands[0]


@ufl_type(is_scalar=True, num_ops=1)
class Determinant(CompoundTensorOperator):
    """Determinant."""

    __slots__ = ()

    def __new__(cls, A):
        """Create new Determinant."""
        sh = A.ufl_shape
        r = len(sh)
        Afi = A.ufl_free_indices

        # Checks
        if r not in (0, 2):
            raise ValueError("Determinant of tensor with rank != 2 is undefined.")
        if r == 2 and sh[0] != sh[1]:
            raise ValueError("Cannot take determinant of rectangular rank 2 tensor.")
        if Afi:
            raise ValueError("Not expecting free indices in determinant.")

        # Simplification
        if isinstance(A, Zero):
            return Zero((), Afi, A.ufl_index_dimensions)
        if r == 0:
            return A

        return CompoundTensorOperator.__new__(cls)

    def __init__(self, A):
        """Initialise."""
        CompoundTensorOperator.__init__(self, (A,))

    def __str__(self):
        """Format as a string."""
        return "det(%s)" % self.ufl_operands[0]


# TODO: Drop Inverse and represent it as product of Determinant and
# Cofactor?
@ufl_type(is_index_free=True, num_ops=1)
class Inverse(CompoundTensorOperator):
    """Inverse."""

    __slots__ = ()

    def __new__(cls, A):
        """Create new Inverse."""
        sh = A.ufl_shape
        r = len(sh)

        # Checks
        if A.ufl_free_indices:
            raise ValueError("Not expecting free indices in Inverse.")
        if isinstance(A, Zero):
            raise ValueError("Division by zero!")

        # Simplification
        if r == 0:
            return 1 / A

        # More checks
        if r != 2:
            raise ValueError("Inverse of tensor with rank != 2 is undefined.")
        if sh[0] != sh[1]:
            raise ValueError(f"Cannot take inverse of rectangular matrix with dimensions {sh}.")

        return CompoundTensorOperator.__new__(cls)

    def __init__(self, A):
        """Initialise."""
        CompoundTensorOperator.__init__(self, (A,))

    @property
    def ufl_shape(self):
        """Get the UFL shape."""
        return self.ufl_operands[0].ufl_shape

    def __str__(self):
        """Format as a string."""
        return "%s^-1" % parstr(self.ufl_operands[0], self)


@ufl_type(is_index_free=True, num_ops=1)
class Cofactor(CompoundTensorOperator):
    """Cofactor."""

    __slots__ = ()

    def __init__(self, A):
        """Initialise."""
        CompoundTensorOperator.__init__(self, (A,))

        # Checks
        sh = A.ufl_shape
        if len(sh) != 2:
            raise ValueError("Cofactor of tensor with rank != 2 is undefined.")
        if sh[0] != sh[1]:
            raise ValueError(f"Cannot take cofactor of rectangular matrix with dimensions {sh}.")
        if A.ufl_free_indices:
            raise ValueError("Not expecting free indices in Cofactor.")
        if isinstance(A, Zero):
            raise ValueError("Cannot take cofactor of zero matrix.")

    @property
    def ufl_shape(self):
        """Get the UFL shape."""
        return self.ufl_operands[0].ufl_shape

    def __str__(self):
        """Format as a string."""
        return "cofactor(%s)" % self.ufl_operands[0]


@ufl_type(num_ops=1, inherit_shape_from_operand=0, inherit_indices_from_operand=0)
class Deviatoric(CompoundTensorOperator):
    """Deviatoric."""

    __slots__ = ()

    def __new__(cls, A):
        """Create new Deviatoric."""
        sh = A.ufl_shape

        # Checks
        if len(sh) != 2:
            raise ValueError("Deviatoric part of tensor with rank != 2 is undefined.")
        if sh[0] != sh[1]:
            raise ValueError(f"Cannot take deviatoric part of rectangular matrix with dimensions {sh}.")
        if A.ufl_free_indices:
            raise ValueError("Not expecting free indices in Deviatoric.")

        # Simplification
        if isinstance(A, Zero):
            return Zero(A.ufl_shape, A.ufl_free_indices, A.ufl_index_dimensions)

        return CompoundTensorOperator.__new__(cls)

    def __init__(self, A):
        """Initialise."""
        CompoundTensorOperator.__init__(self, (A,))

    def __str__(self):
        """Format as a string."""
        return "dev(%s)" % self.ufl_operands[0]


@ufl_type(num_ops=1, inherit_shape_from_operand=0, inherit_indices_from_operand=0)
class Skew(CompoundTensorOperator):
    """Skew."""

    __slots__ = ()

    def __new__(cls, A):
        """Create new Skew."""
        sh = A.ufl_shape
        Afi = A.ufl_free_indices

        # Checks
        if len(sh) != 2:
            raise ValueError("Skew symmetric part of tensor with rank != 2 is undefined.")
        if sh[0] != sh[1]:
            raise ValueError(f"Cannot take skew part of rectangular matrix with dimensions {sh}.")
        if Afi:
            raise ValueError("Not expecting free indices in Skew.")

        # Simplification
        if isinstance(A, Zero):
            return Zero(A.ufl_shape, Afi, A.ufl_index_dimensions)

        return CompoundTensorOperator.__new__(cls)

    def __init__(self, A):
        """Initialise."""
        CompoundTensorOperator.__init__(self, (A,))

    def __str__(self):
        """Format as a string."""
        return "skew(%s)" % self.ufl_operands[0]


@ufl_type(num_ops=1, inherit_shape_from_operand=0, inherit_indices_from_operand=0)
class Sym(CompoundTensorOperator):
    """Sym."""

    __slots__ = ()

    def __new__(cls, A):
        """Create new Sym."""
        sh = A.ufl_shape
        Afi = A.ufl_free_indices

        # Checks
        if len(sh) != 2:
            raise ValueError("Symmetric part of tensor with rank != 2 is undefined.")
        if sh[0] != sh[1]:
            raise ValueError(f"Cannot take symmetric part of rectangular matrix with dimensions {sh}.")
        if Afi:
            raise ValueError("Not expecting free indices in Sym.")

        # Simplification
        if isinstance(A, Zero):
            return Zero(A.ufl_shape, Afi, A.ufl_index_dimensions)

        return CompoundTensorOperator.__new__(cls)

    def __init__(self, A):
        """Initialise."""
        CompoundTensorOperator.__init__(self, (A,))

    def __str__(self):
        """Format as a string."""
        return f"sym({self.ufl_operands[0]})"
