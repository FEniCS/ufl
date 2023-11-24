"""This module defines the ``Expr`` class, the superclass for all expression tree node types in UFL."""
# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2008
# Modified by Massimiliano Leoni, 2016

import numbers
import typing
import warnings
from abc import abstractmethod, abstractproperty

from ufl.core.ufl_type import UFLObject


class Expr(UFLObject):
    """Base class for all UFL expression types.

    *Instance properties*
        Every ``Expr`` instance will have certain properties.
        The most important ones are ``ufl_operands``, ``ufl_shape``,
        ``ufl_free_indices``, and ``ufl_index_dimensions`` properties.
        Expressions are immutable and hashable.

    *Type traits*
        The ``Expr`` API defines a number of type traits that each subclass
        needs to provide. Most of these are specified indirectly via
        the arguments to the ``ufl_type`` class decorator, allowing UFL
        to do some consistency checks and automate most of the traits
        for most types. Type traits are accessed via a class or
        instance object of the form ``obj._ufl_traitname_``. See the source
        code for description of each type trait.

    *Operators*
        Some Python special functions are implemented in this class,
        some are implemented in subclasses, and some are attached to
        this class in the ``ufl_type`` class decorator.

    *Defining subclasses*
        To define a new expression class, inherit from either
        ``Terminal`` or ``Operator``, and apply the ``ufl_type`` class
        decorator with suitable arguments.  See the docstring of
        ``ufl_type`` for details on its arguments.  Looking at existing
        classes similar to the one you wish to add is a good
        idea. Looking through the comments in the ``Expr`` class and
        ``ufl_type`` to understand all the properties that may need to
        be specified is also a good idea. Note that many algorithms in
        UFL and form compilers will need handlers implemented for each
        new type::.

        .. code-block:: python

            class MyOperator(UFLObject):
                pass

    *Type collections*
        All ``Expr`` subclasses are collected by ``ufl_type`` in global
        variables available via ``Expr``.

    *Profiling*
        Object creation statistics can be collected by doing

        .. code-block:: python

            Expr.ufl_enable_profiling()
            # ... run some code
            initstats, delstats = Expr.ufl_disable_profiling()

        Giving a list of creation and deletion counts for each typecode.
    """

    # --- Each Expr subclass must define __slots__ or _ufl_noslots_ at
    # --- the top ---
    # This is to freeze member variables for objects of this class and
    # save memory by skipping the per-instance dict.

    __slots__ = ("_hash", )
    # _ufl_noslots_ = True

    # --- Basic object behaviour ---

    def __getnewargs__(self):
        """Get newargs tuple.

        The tuple returned here is passed to as args to cls.__new__(cls, *args).

        This implementation passes the operands, which is () for terminals.

        May be necessary to override if __new__ is implemented in a subclass.
        """
        return self.ufl_operands

    # This shows the principal behaviour of the hash function attached
    # in ufl_type:
    # def __hash__(self):
    #     if self._hash is None:
    #         self._hash = self._ufl_compute_hash_()
    #     return self._hash

    # --- Type traits are added to subclasses by the ufl_type class
    # --- decorator ---

    # Note: Some of these are modified after the Expr class definition
    # because Expr is not defined yet at this point.  Note: Boolean
    # type traits that categorize types are mostly set to None for
    # Expr but should be True or False for any non-abstract type.

    # A reference to the UFL class itself.  This makes it possible to
    # do type(f)._ufl_class_ and be sure you get the actual UFL class
    # instead of a subclass from another library.
    _ufl_class_: typing.Optional[type] = None

    # The handler name.  This is the name of the handler function you
    # implement for this type in a multifunction.
    _ufl_handler_name_ = "expr"

    # Number of operands, "varying" for some types, or None if not
    # applicable for abstract types.
    _ufl_num_ops_ = None

    # Type trait: If the type is a literal.
    _ufl_is_literal_ = None

    # Type trait: If the type is classified as a 'terminal modifier',
    # for form compiler use.
    _ufl_is_terminal_modifier_ = None

    # Type trait: If the type is a shaping operator.  Shaping
    # operations include indexing, slicing, transposing, i.e. not
    # introducing computation of a new value.
    _ufl_is_shaping_ = False

    # Type trait: If the type is in reference frame.
    _ufl_is_in_reference_frame_ = None

    # Type trait: If the type is a restriction to a geometric entity.
    _ufl_is_restriction_ = None

    # Type trait: If the type is evaluation in a particular way.
    _ufl_is_evaluation_ = None

    # Type trait: If the type is a differential operator.
    _ufl_is_differential_ = None

    # Type trait: If the type is purely scalar, having no shape or
    # indices.
    _ufl_is_scalar_ = None

    # Type trait: If the type never has free indices.
    _ufl_is_index_free_ = False

    # --- Global variables for collecting all types ---

    # A global dict mapping language_operator_name to the type it
    # produces
    _ufl_language_operators_: typing.Dict[str, type] = {}

    # List of all terminal modifier types
    _ufl_terminal_modifiers_: typing.List[type] = []

    # --- Mechanism for profiling object creation and deletion ---

    @staticmethod
    def ufl_enable_profiling():
        """Turn on the object counting mechanism and reset counts to zero."""
        Expr.__init__ = Expr._ufl_profiling__init__
        setattr(Expr, "__del__", Expr._ufl_profiling__del__)
        for i in range(len(Expr._ufl_obj_init_counts_)):
            Expr._ufl_obj_init_counts_[i] = 0
            Expr._ufl_obj_del_counts_[i] = 0

    @staticmethod
    def ufl_disable_profiling():
        """Turn off the object counting mechanism. Return object init and del counts."""
        Expr.__init__ = Expr._ufl_regular__init__
        delattr(Expr, "__del__")
        return (Expr._ufl_obj_init_counts_, Expr._ufl_obj_del_counts_)

    # TODO: fix UFL typing and make the following actual abstract methods and properties

    @abstractproperty
    def ufl_operands(self):
        """A tuple of operands, all of them Expr instances."""

    @abstractproperty
    def ufl_shape(self):
        """A tuple of ints, the value shape of the expression."""
        raise NotImplementedError()

    @abstractproperty
    def ufl_free_indices(self):
        """A tuple of free index counts."""

    @abstractproperty
    def ufl_index_dimensions(self):
        """A tuple providing the int dimension for each free index."""

    @abstractmethod
    def _ufl_compute_hash_(self):
        """To compute the hash on demand, this method is called."""

    def _ufl_expr_reconstruct_(self, *operands):
        """Return a new object of the same type with new operands."""
        raise NotImplementedError()

    @abstractmethod
    def _ufl_signature_data_(self, renumbering):
        """Return data that uniquely identifies form compiler relevant aspects of this object."""

    @abstractmethod
    def __repr__(self):
        """Return string representation this object can be reconstructed from."""

    @abstractmethod
    def __str__(self):
        """Return pretty print string representation of this object."""

    def ufl_domains(self):
        """Return all domains this expression is defined on."""
        warnings.warn("Expr.ufl_domains() is deprecated, please "
                      "use extract_domains(expr) instead.", DeprecationWarning)
        from ufl.domain import extract_domains
        return extract_domains(self)

    def ufl_domain(self):
        """Return the single unique domain this expression is defined on, or throw an error."""
        warnings.warn("Expr.ufl_domain() is deprecated, please "
                      "use extract_unique_domain(expr) instead.", DeprecationWarning)
        from ufl.domain import extract_unique_domain
        return extract_unique_domain(self)

    def evaluate(self, x, mapping, component, index_values):
        """Evaluate expression at given coordinate with given values for terminals."""
        raise ValueError(f"Symbolic evaluation of {self._ufl_class_.__name__} not available.")

    def _ufl_evaluate_scalar_(self):
        if self.ufl_shape or self.ufl_free_indices:
            raise TypeError("Cannot evaluate a nonscalar expression to a scalar value.")
        return self(())  # No known x

    def __float__(self):
        """Try to evaluate as scalar and cast to float."""
        try:
            v = float(self._ufl_evaluate_scalar_())
        except Exception:
            v = NotImplemented
        return v

    def __complex__(self):
        """Try to evaluate as scalar and cast to complex."""
        try:
            v = complex(self._ufl_evaluate_scalar_())
        except TypeError:
            v = NotImplemented
        return v

    def __bool__(self):
        """By default, all Expr are nonzero/False."""
        return True

    def __nonzero__(self):
        """By default, all Expr are nonzero/False."""
        return self.__bool__()

    @staticmethod
    def _ufl_coerce_(value):
        """Convert any value to a UFL type."""
        # Quick skip for most types
        if isinstance(value, Expr):
            return value

        # Conversion from non-ufl types
        # (the _ufl_from_*_ functions are attached to Expr by ufl_type)
        ufl_from_type = "_ufl_from_{0}_".format(value.__class__.__name__)
        return getattr(Expr, ufl_from_type)(value)

    def _ufl_err_str_(self):
        """Return a short string to represent this Expr in an error message."""
        return f"<{self._ufl_class_.__name__} id={id(self)}>"

    def __eq__(self, other):
        """Checks whether the two expressions are represented the exact same way.

        This does not check if the expressions are
        mathematically equal or equivalent! Used by sets and dicts.
        """
        # Fast cutoffs for common cases, type difference or hash
        # difference will cutoff more or less all nonequal types
        if type(self) is not type(other) or hash(self) != hash(other):
            return False

        # Large objects are costly to compare with themselves
        if self is other:
            return True

        # Modelled after pre_traversal to avoid recursion:
        left = [(self, other)]
        while left:
            s, o = left.pop()

            if s._is_terminal():
                # Compare terminals
                if not s == o:
                    return False
            else:
                # Delve into subtrees
                so = s.ufl_operands
                oo = o.ufl_operands
                if len(so) != len(oo):
                    return False

                for s, o in zip(so, oo):
                    # Fast cutoff for common case
                    if s._typecode() != o._typecode():
                        return False
                    # Skip subtree if objects are the same
                    if s is o:
                        continue
                    # Append subtree for further inspection
                    left.append((s, o))

        # Equal if we get out of the above loop!
        # Eagerly DAGify to reduce the size of the tree.
        self.ufl_operands = other.ufl_operands
        return True

    def __ne__(self, other):
        """Check if not equal."""
        return not self.__eq__(other)

    def __le__(self, other):
        """Return less than or equal conditional."""
        from ufl.conditional import LE
        return LE(self, other)

    def __ge__(self, other):
        """Return greater than or equal conditional."""
        from ufl.conditional import GE
        return GE(self, other)

    def ___lt__(self, other):
        """Return less than conditional."""
        from ufl.conditional import LT
        return LT(self, other)

    def __gt__(self, other):
        """Return greater than conditional."""
        from ufl.conditional import GT
        return GT(self, other)

    def __xor__(self, indices):
        """A^indices := as_tensor(A, indices)."""
        from ufl.core.multiindex import Index
        from ufl.tensors import as_tensor

        if not isinstance(indices, tuple):
            raise ValueError("Expecting a tuple of Index objects to A^indices := as_tensor(A, indices).")
        if not all(isinstance(i, Index) for i in indices):
            raise ValueError("Expecting a tuple of Index objects to A^indices := as_tensor(A, indices).")
        return as_tensor(self, indices)

    def __mul__(self, other):
        """Multiply."""
        from ufl.algebra import Product
        from ufl.constantvalue import Zero, as_ufl
        from ufl.core.multiindex import Index, MultiIndex, indices
        from ufl.index_combination_utils import merge_overlapping_indices
        from ufl.indexsum import IndexSum
        from ufl.tensors import as_tensor

        if not isinstance(other, (Expr, numbers.Real, numbers.Integral, numbers.Complex)):
            return NotImplemented
        other = as_ufl(other)
        # Discover repeated indices, which results in index sums
        afi = self.ufl_free_indices
        bfi = other.ufl_free_indices
        afid = self.ufl_index_dimensions
        bfid = other.ufl_index_dimensions
        fi, fid, ri, rid = merge_overlapping_indices(afi, afid, bfi, bfid)

        # Pick out valid non-scalar products here (dot products):
        # - matrix-matrix (A*B, M*grad(u)) => A . B
        # - matrix-vector (A*v) => A . v
        s1, s2 = self.ufl_shape, other.ufl_shape
        r1, r2 = len(s1), len(s2)

        if r1 == 0 and r2 == 0:
            # Create scalar product
            p = Product(self, other)
            ti = ()

        elif r1 == 0 or r2 == 0:
            # Scalar - tensor product
            if r2 == 0:
                self, other = other, self

            # Check for zero, simplifying early if possible
            if isinstance(self, Zero) or isinstance(other, Zero):
                shape = s1 or s2
                return Zero(shape, fi, fid)

            # Repeated indices are allowed, like in:
            # v[i]*M[i,:]

            # Apply product to scalar components
            ti = indices(len(other.ufl_shape))
            p = Product(self, other[ti])

        elif r1 == 2 and r2 in (1, 2):  # Matrix-matrix or matrix-vector
            if ri:
                raise ValueError("Not expecting repeated indices in non-scalar product.")

            # Check for zero, simplifying early if possible
            if isinstance(self, Zero) or isinstance(other, Zero):
                shape = s1[:-1] + s2[1:]
                return Zero(shape, fi, fid)

            # Return dot product in index notation
            ai = indices(len(self.ufl_shape) - 1)
            bi = indices(len(other.ufl_shape) - 1)
            k = indices(1)

            p = self[ai + k] * other[k + bi]
            ti = ai + bi

        else:
            raise ValueError(f"Invalid ranks {r1} and {r2} in product.")

        # TODO: I think applying as_tensor after index sums results in
        # cleaner expression graphs.
        # Wrap as tensor again
        if ti:
            p = as_tensor(p, ti)

        # If any repeated indices were found, apply implicit summation
        # over those
        for i in ri:
            mi = MultiIndex((Index(count=i),))
            p = IndexSum(p, mi)

        return p

    def __rmul__(self, other):
        """Multiply."""
        from ufl.constantvalue import as_ufl
        if not isinstance(other, (Expr, numbers.Real, numbers.Integral, numbers.Complex)):
            return NotImplemented
        other = as_ufl(other)
        return other.__mul__(self)

    def __add__(self, other):
        """Add."""
        from ufl.algebra import Sum
        if not isinstance(other, (Expr, numbers.Real, numbers.Integral, numbers.Complex)):
            return NotImplemented
        return Sum(self, other)

    def __radd__(self, other):
        """Add."""
        from ufl.algebra import Sum
        if not isinstance(other, (Expr, numbers.Real, numbers.Integral, numbers.Complex)):
            return NotImplemented
        if isinstance(other, numbers.Number) and other == 0:
            return self
        return Sum(other, self)

    def __sub__(self, other):
        """Subtract."""
        from ufl.algebra import Sum
        if not isinstance(other, (Expr, numbers.Real, numbers.Integral, numbers.Complex)):
            return NotImplemented
        return Sum(self, -other)

    def __rsub__(self, other):
        """Subtract."""
        from ufl.algebra import Sum
        if not isinstance(other, (Expr, numbers.Real, numbers.Integral, numbers.Complex)):
            return NotImplemented
        return Sum(other, -self)

    def __div__(self, other):
        """Divide."""
        from ufl.algebra import Division
        from ufl.core.multiindex import indices
        from ufl.tensors import as_tensor

        if not isinstance(other, (Expr, numbers.Real, numbers.Integral, numbers.Complex)):
            return NotImplemented
        sh = self.ufl_shape
        if sh:
            ii = indices(len(sh))
            d = Division(self[ii], other)
            return as_tensor(d, ii)
        return Division(self, other)

    def __truediv__(self, other):
        """Divide."""
        return self.__div__(other)

    def __rdiv__(self, other):
        """Divide."""
        from ufl.algebra import Division
        if not isinstance(other, (Expr, numbers.Real, numbers.Integral, numbers.Complex)):
            return NotImplemented
        return Division(other, self)

    def __rtruediv__(self, other):
        """Divide."""
        return self.__rdiv__(other)

    def __pow__(self, other):
        """Raise to a power."""
        from ufl.algebra import Power
        from ufl.tensoralgebra import Inner

        if not isinstance(other, (Expr, numbers.Real, numbers.Integral, numbers.Complex)):
            return NotImplemented
        if other == 2 and self.ufl_shape:
            return Inner(self, self)
        return Power(self, other)

    def __rpow__(self, other):
        """Raise to a power."""
        from ufl.algebra import Power

        if not isinstance(other, (Expr, numbers.Real, numbers.Integral, numbers.Complex)):
            return NotImplemented
        return Power(other, self)

    def __neg__(self):
        """Negate."""
        return -1 * self

    def __abs__(self):
        """Absolute value."""
        from ufl.algebra import Abs
        return Abs(self)

    def __call__(self, arg, mapping=None, component=()):
        """Take a restriction or evaluate depending on argument."""
        from ufl.restriction import NegativeRestricted, PositiveRestricted
        from ufl.utils.stacks import StackDict

        if arg in ("+", "-"):
            if mapping is not None:
                raise ValueError("Not expecting a mapping when taking restriction.")
            if arg == "+":
                return PositiveRestricted(self)
            if arg == "-":
                return NegativeRestricted(self)
            raise ValueError(f"Invalid side '{arg}' in restriction operator.")
        else:
            # Evaluate derivatives first
            from ufl.algorithms import expand_derivatives
            f = expand_derivatives(self)

            # Evaluate recursively
            if mapping is None:
                mapping = {}
            index_values = StackDict()
            return f.evaluate(arg, mapping, component, index_values)

    def __getitem__(self, component):
        """Get an item."""
        from ufl.constantvalue import Zero
        from ufl.core.multiindex import MultiIndex
        from ufl.index_combination_utils import create_slice_indices
        from ufl.indexed import Indexed
        from ufl.indexsum import IndexSum
        from ufl.tensors import ComponentTensor, as_tensor

        # Treat component consistently as tuple below
        if not isinstance(component, tuple):
            component = (component,)

        shape = self.ufl_shape

        # Analyse slices (:) and Ellipsis (...)
        all_indices, slice_indices, repeated_indices = create_slice_indices(component, shape, self.ufl_free_indices)

        # Check that we have the right number of indices for a tensor with
        # this shape
        if len(shape) != len(all_indices):
            raise ValueError(f"Invalid number of indices {len(all_indices)} for expression of rank {len(shape)}.")

        # Special case for simplifying foo[...] => foo, foo[:] => foo or
        # similar
        if len(slice_indices) == len(all_indices):
            return self

        # Special case for simplifying as_tensor(ai,(i,))[i] => ai
        if isinstance(self, ComponentTensor):
            if all_indices == self.indices().indices():
                return self.ufl_operands[0]

        # Apply all indices to index self, yielding a scalar valued
        # expression
        mi = MultiIndex(all_indices)
        a = Indexed(self, mi)

        # TODO: I think applying as_tensor after index sums results in
        # cleaner expression graphs.

        # If the Ellipsis or any slices were found, wrap as tensor valued
        # with the slice indices created at the top here
        if slice_indices:
            a = as_tensor(a, slice_indices)

        # If any repeated indices were found, apply implicit summation
        # over those
        for i in repeated_indices:
            mi = MultiIndex((i,))
            a = IndexSum(a, mi)

        # Check for zero (last so we can get indices etc from a, could
        # possibly be done faster by checking early instead)
        if isinstance(self, Zero):
            shape = a.ufl_shape
            fi = a.ufl_free_indices
            fid = a.ufl_index_dimensions
            a = Zero(shape, fi, fid)

        return a

    def __len__(self):
        """Length of expression. Used for iteration over vector expressions."""
        s = self.ufl_shape
        if len(s) == 1:
            return s[0]
        raise NotImplementedError("Cannot take length of non-vector expression.")

    def __iter__(self):
        """Iteration over vector expressions."""
        for i in range(len(self)):
            yield self[i]

    def __floordiv__(self, other):
        """UFL does not support integer division."""
        raise NotImplementedError(self.__class__.__floordiv__)

    def __pos__(self):
        """Unary + is a no-op."""
        return self

    def __round__(self, n=None):
        """Round to nearest integer or to nearest nth decimal."""
        try:
            val = float(self._ufl_evaluate_scalar_())
            val = round(val, n)
        except TypeError:
            val = complex(self._ufl_evaluate_scalar_())
            val = round(val.real, n) + round(val.imag, n) * 1j
        except TypeError:
            val = NotImplemented
        return val

    def dx(self, *ii):
        """Return the partial derivative with respect to spatial variable number *ii*."""
        from ufl.differentiation import Grad

        d = self
        # Unwrap ii to allow .dx(i,j) and .dx((i,j))
        if len(ii) == 1 and isinstance(ii[0], tuple):
            ii = ii[0]
        # Apply all derivatives
        for i in ii:
            d = Grad(d)

        # Take all components, applying repeated index sums in the [] operation
        return d.__getitem__((Ellipsis,) + ii)

    @property
    def T(self):
        """Transpose a rank-2 tensor expression.

        For more general transpose operations of higher order tensor expressions, use indexing and Tensor.
        """
        from ufl.tensoralgebra import Transposed
        return Transposed(self)


# Initializing traits here because Expr is not defined in the class
# declaration
Expr._ufl_class_ = Expr


def ufl_err_str(expr):
    """Return a UFL error string."""
    if hasattr(expr, "_ufl_err_str_"):
        return expr._ufl_err_str_()
    else:
        return repr(expr)
