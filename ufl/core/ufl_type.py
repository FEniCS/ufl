"""UFL type."""
# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Massimiliano Leoni, 2016
# Modified by Matthew Scroggs, 2023

from __future__ import annotations

import inspect
import typing
from abc import ABC, abstractmethod

import ufl.core as core
from ufl.core.compute_expr_hash import compute_expr_hash
from ufl.utils.formatting import camel2underscore

all_ufl_classes = []


class UFLObject(ABC):
    """A UFL Object."""
    __slots__ = ()
    # A global counter of the number of typecodes assigned.
    _ufl_num_typecodes_ = 0

    # Set the handler name for UFLType
    _ufl_handler_name_ = "ufl_type"

    # A global set of all handler names added
    _ufl_all_handler_names_: typing.Set[str] = set()

    # A global array of the number of initialized objects for each
    # typecode
    _ufl_obj_init_counts_: typing.List[int] = []

    # A global array of the number of deleted objects for each
    # typecode
    _ufl_obj_del_counts_: typing.List[int] = []

    @classmethod
    def _typecode(cls) -> int:
        global all_ufl_classes
        if cls not in all_ufl_classes:
            all_ufl_classes.append(cls)
        return all_ufl_classes.index(cls)

    @classmethod
    def _is_terminal(cls) -> bool:
        return False

    @abstractmethod
    def _ufl_hash_data_(self) -> typing.Hashable:
        """Return hashable data that uniquely defines this object."""

    @abstractmethod
    def __str__(self) -> str:
        """Return a human-readable string representation of the object."""

    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of the object."""

    def __hash__(self) -> int:
        """Hash the object."""
        return hash(self._ufl_hash_data_())

    def __eq__(self, other):
        """Check if two objects are equal."""
        return type(self) is type(other) and self._ufl_hash_data_() == other._ufl_hash_data_()

    def __ne__(self, other):
        """Check inequality."""
        return not self.__eq__(other)


def get_base_attr(cls, name):
    """Return first non-``None`` attribute of given name among base classes."""
    for base in cls.mro():
        if hasattr(base, name):
            attr = getattr(base, name)
            if attr is not None:
                return attr
    return None


def set_trait(cls, basename, value, inherit=False):
    """Assign a trait to class with namespacing ``_ufl_basename_`` applied.

    If trait value is ``None``, optionally inherit it from the closest base class that has it.
    """
    name = "_ufl_" + basename + "_"
    if value is None and inherit:
        value = get_base_attr(cls, name)
    setattr(cls, name, value)


def determine_num_ops(cls, num_ops, unop, binop, rbinop):
    """Determine number of operands for this type."""
    # Try to determine num_ops from other traits or baseclass, or
    # require num_ops to be set for non-abstract classes if it cannot
    # be determined automatically
    if num_ops is not None:
        return num_ops
    elif cls._is_terminal():
        return 0
    elif unop:
        return 1
    elif binop or rbinop:
        return 2
    else:
        # Determine from base class
        return get_base_attr(cls, "_ufl_num_ops_")


def check_is_terminal_consistency(cls):
    """Check for consistency in ``is_terminal`` trait among superclasses."""
    if cls._is_terminal() is None:
        msg = (f"Class {cls.__name__} has not specified the is_terminal trait."
               " Did you forget to inherit from Terminal or Operator?")
        raise TypeError(msg)

    base_is_terminal = get_base_attr(cls, "_is_terminal")()
    if base_is_terminal is not None and cls._is_terminal() != base_is_terminal:
        msg = (f"Conflicting given and automatic 'is_terminal' trait for class {cls.__name__}."
               " Check if you meant to inherit from Terminal or Operator.")
        raise TypeError(msg)


def check_abstract_trait_consistency(cls):
    """Check that the first base classes up to ``Expr`` are other UFL types."""
    for base in cls.mro():
        if base is core.expr.Expr:
            break
        if not issubclass(base, core.expr.Expr) and inspect.isabstract(base):
            msg = ("Base class {0.__name__} of class {1.__name__} "
                   "is not an abstract subclass of {2.__name__}.")
            raise TypeError(msg.format(base, cls, core.expr.Expr))


def check_has_slots(cls):
    """Check if type has __slots__ unless it is marked as exception with _ufl_noslots_."""
    if "_ufl_noslots_" in cls.__dict__:
        return

    if "__slots__" not in cls.__dict__:
        msg = ("Class {0.__name__} is missing the __slots__ "
               "attribute and is not marked with _ufl_noslots_.")
        raise TypeError(msg.format(cls))

    # Check base classes for __slots__ as well, skipping object which is the last one
    for base in cls.mro()[1:-1]:
        if "__slots__" not in base.__dict__:
            msg = ("Class {0.__name__} is has a base class "
                   "{1.__name__} with __slots__ missing.")
            raise TypeError(msg.format(cls, base))


def check_type_traits_consistency(cls):
    """Execute a variety of consistency checks on the ufl type traits."""
    # Check for consistency in global type collection sizes
    Expr = core.expr.Expr
    assert Expr._ufl_num_typecodes_ == len(Expr._ufl_all_handler_names_)
    assert Expr._ufl_num_typecodes_ == len(Expr._ufl_all_classes_)
    assert Expr._ufl_num_typecodes_ == len(Expr._ufl_obj_init_counts_)
    assert Expr._ufl_num_typecodes_ == len(Expr._ufl_obj_del_counts_)

    # Check that non-abstract types always specify num_ops
    #if not inspect.isabstract(cls):
    #    if cls._ufl_num_ops_ is None:
    #        msg = "Class {0.__name__} has not specified num_ops."
    #        raise TypeError(msg.format(cls))

    # Check for non-abstract types that num_ops has the right type
    #if not inspect.isabstract(cls):
    #    if not (isinstance(cls._ufl_num_ops_, int) or cls._ufl_num_ops_ == "varying"):
    #        msg = 'Class {0.__name__} has invalid num_ops value {1} (integer or "varying").'
    #        raise TypeError(msg.format(cls, cls._ufl_num_ops_))

    # Check that num_ops is not set to nonzero for a terminal
    #if cls._is_terminal() and cls._ufl_num_ops_ != 0:
    #    msg = "Class {0.__name__} has num_ops > 0 but is terminal."
    #    raise TypeError(msg.format(cls))

    # Check that a non-scalar type doesn't have a scalar base class.
    #if not cls._ufl_is_scalar_:
    #    if get_base_attr(cls, "_ufl_is_scalar_"):
    #        msg = "Non-scalar class {0.__name__} is has a scalar base class."
    #        raise TypeError(msg.format(cls))


def attach_implementations_of_indexing_interface(
    cls, inherit_shape_from_operand, inherit_indices_from_operand
):
    """Attach implementations of indexing interface."""
    # Scalar or index-free? Then we can simplify the implementation of
    # tensor properties by attaching them here.
    if cls._ufl_is_scalar_:
        cls.ufl_shape = ()

    if cls._ufl_is_scalar_ or cls._ufl_is_index_free_:
        cls.ufl_free_indices = ()
        cls.ufl_index_dimensions = ()

    # Automate direct inheriting of shape and indices from one of the
    # operands.  This simplifies refactoring because a lot of types do
    # this.
    if inherit_shape_from_operand is not None:
        def _inherited_ufl_shape(self):
            return self.ufl_operands[inherit_shape_from_operand].ufl_shape
        cls.ufl_shape = property(_inherited_ufl_shape)

    if inherit_indices_from_operand is not None:
        def _inherited_ufl_free_indices(self):
            return self.ufl_operands[inherit_indices_from_operand].ufl_free_indices

        def _inherited_ufl_index_dimensions(self):
            return self.ufl_operands[inherit_indices_from_operand].ufl_index_dimensions
        cls.ufl_free_indices = property(_inherited_ufl_free_indices)
        cls.ufl_index_dimensions = property(_inherited_ufl_index_dimensions)


def update_global_expr_attributes(cls):
    """Update global ``Expr`` attributes, mainly by adding *cls* to global collections of ufl types."""
    if cls._ufl_is_terminal_modifier_:
        core.expr.Expr._ufl_terminal_modifiers_.append(cls)

    # Add to collection of language operators.  This collection is
    # used later to populate the official language namespace.
    # TODO: I don't think this functionality is fully completed, check
    # it out later.
    if not inspect.isabstract(cls) and hasattr(cls, "_ufl_function_"):
        cls._ufl_function_.__func__.__doc__ = cls.__doc__
        core.expr.Expr._ufl_language_operators_[cls._ufl_handler_name_] = cls._ufl_function_
