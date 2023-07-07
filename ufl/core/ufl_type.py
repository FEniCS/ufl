# -*- coding: utf-8 -*-

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Massimiliano Leoni, 2016
# Modified by Matthew Scroggs, 2023

from __future__ import annotations
import typing
import warnings

from ufl.core.compute_expr_hash import compute_expr_hash
from ufl.utils.formatting import camel2underscore
from abc import ABC, abstractmethod
# Avoid circular import
import ufl.core as core


class UFLObject(ABC):
    """A UFL Object."""

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
        return hash(self._ufl_hash_data_())

    def __eq__(self, other):
        return type(self) == type(other) and self._ufl_hash_data_() == other._ufl_hash_data_()

    def __ne__(self, other):
        return not self.__eq__(other)


def attach_operators_from_hash_data(cls):
    """Class decorator to attach ``__hash__``, ``__eq__`` and ``__ne__`` implementations.

    These are implemented in terms of a ``._ufl_hash_data()`` method on the class,
    which should return a tuple or hashable and comparable data.
    """
    warnings.warn("attach_operators_from_hash_data deprecated, please use UFLObject instead.", DeprecationWarning)
    assert hasattr(cls, "_ufl_hash_data_")

    def __hash__(self):
        "__hash__ implementation attached in attach_operators_from_hash_data"
        return hash(self._ufl_hash_data_())
    cls.__hash__ = __hash__

    def __eq__(self, other):
        "__eq__ implementation attached in attach_operators_from_hash_data"
        return type(self) == type(other) and self._ufl_hash_data_() == other._ufl_hash_data_()
    cls.__eq__ = __eq__

    def __ne__(self, other):
        "__ne__ implementation attached in attach_operators_from_hash_data"
        return not self.__eq__(other)
    cls.__ne__ = __ne__

    return cls


def get_base_attr(cls, name):
    "Return first non-``None`` attribute of given name among base classes."
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
    "Determine number of operands for this type."
    # Try to determine num_ops from other traits or baseclass, or
    # require num_ops to be set for non-abstract classes if it cannot
    # be determined automatically
    if num_ops is not None:
        return num_ops
    elif cls._ufl_is_terminal_:
        return 0
    elif unop:
        return 1
    elif binop or rbinop:
        return 2
    else:
        # Determine from base class
        return get_base_attr(cls, "_ufl_num_ops_")


def check_is_terminal_consistency(cls):
    "Check for consistency in ``is_terminal`` trait among superclasses."
    if cls._ufl_is_terminal_ is None:
        msg = ("Class {0.__name__} has not specified the is_terminal trait." +
               " Did you forget to inherit from Terminal or Operator?")
        raise TypeError(msg.format(cls))

    base_is_terminal = get_base_attr(cls, "_ufl_is_terminal_")
    if base_is_terminal is not None and cls._ufl_is_terminal_ != base_is_terminal:
        msg = ("Conflicting given and automatic 'is_terminal' trait for class {0.__name__}." +
               " Check if you meant to inherit from Terminal or Operator.")
        raise TypeError(msg.format(cls))


def check_abstract_trait_consistency(cls):
    "Check that the first base classes up to ``Expr`` are other UFL types."
    for base in cls.mro():
        if base is core.expr.Expr:
            break
        if not issubclass(base, core.expr.Expr) and base._ufl_is_abstract_:
            msg = ("Base class {0.__name__} of class {1.__name__} "
                   "is not an abstract subclass of {2.__name__}.")
            raise TypeError(msg.format(base, cls, core.expr.Expr))


def check_has_slots(cls):
    """Check if type has ``__slots__`` unless it is marked as exception with
    ``_ufl_noslots_``."""
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
    "Execute a variety of consistency checks on the ufl type traits."

    # Check for consistency in global type collection sizes
    Expr = core.expr.Expr
    assert Expr._ufl_num_typecodes_ == len(Expr._ufl_all_handler_names_)
    assert Expr._ufl_num_typecodes_ == len(Expr._ufl_all_classes_)
    assert Expr._ufl_num_typecodes_ == len(Expr._ufl_obj_init_counts_)
    assert Expr._ufl_num_typecodes_ == len(Expr._ufl_obj_del_counts_)

    # Check that non-abstract types always specify num_ops
    if not cls._ufl_is_abstract_:
        if cls._ufl_num_ops_ is None:
            msg = "Class {0.__name__} has not specified num_ops."
            raise TypeError(msg.format(cls))

    # Check for non-abstract types that num_ops has the right type
    if not cls._ufl_is_abstract_:
        if not (isinstance(cls._ufl_num_ops_, int) or cls._ufl_num_ops_ == "varying"):
            msg = 'Class {0.__name__} has invalid num_ops value {1} (integer or "varying").'
            raise TypeError(msg.format(cls, cls._ufl_num_ops_))

    # Check that num_ops is not set to nonzero for a terminal
    if cls._ufl_is_terminal_ and cls._ufl_num_ops_ != 0:
        msg = "Class {0.__name__} has num_ops > 0 but is terminal."
        raise TypeError(msg.format(cls))

    # Check that a non-scalar type doesn't have a scalar base class.
    if not cls._ufl_is_scalar_:
        if get_base_attr(cls, "_ufl_is_scalar_"):
            msg = "Non-scalar class {0.__name__} is has a scalar base class."
            raise TypeError(msg.format(cls))


def check_implements_required_methods(cls):
    """Check if type implements the required methods."""
    if not cls._ufl_is_abstract_:
        for attr in core.expr.Expr._ufl_required_methods_:
            if not hasattr(cls, attr):
                msg = "Class {0.__name__} has no {1} method."
                raise TypeError(msg.format(cls, attr))
            elif not callable(getattr(cls, attr)):
                msg = "Required method {1} of class {0.__name__} is not callable."
                raise TypeError(msg.format(cls, attr))


def check_implements_required_properties(cls):
    "Check if type implements the required properties."
    if not cls._ufl_is_abstract_:
        for attr in core.expr.Expr._ufl_required_properties_:
            if not hasattr(cls, attr):
                msg = "Class {0.__name__} has no {1} property."
                raise TypeError(msg.format(cls, attr))
            elif callable(getattr(cls, attr)):
                msg = "Required property {1} of class {0.__name__} is a callable method."
                raise TypeError(msg.format(cls, attr))


def attach_implementations_of_indexing_interface(cls,
                                                 inherit_shape_from_operand,
                                                 inherit_indices_from_operand):
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
    "Update global ``Expr`` attributes, mainly by adding *cls* to global collections of ufl types."
    if cls._ufl_is_terminal_modifier_:
        core.expr.Expr._ufl_terminal_modifiers_.append(cls)

    # Add to collection of language operators.  This collection is
    # used later to populate the official language namespace.
    # TODO: I don't think this functionality is fully completed, check
    # it out later.
    if not cls._ufl_is_abstract_ and hasattr(cls, "_ufl_function_"):
        cls._ufl_function_.__func__.__doc__ = cls.__doc__
        core.expr.Expr._ufl_language_operators_[cls._ufl_handler_name_] = cls._ufl_function_


def update_ufl_type_attributes(cls):
    # Determine integer typecode by incrementally counting all types
    cls._ufl_typecode_ = UFLType._ufl_num_typecodes_
    UFLType._ufl_num_typecodes_ += 1

    UFLType._ufl_all_classes_.append(cls)

    # Determine handler name by a mapping from "TypeName" to "type_name"
    cls._ufl_handler_name_ = camel2underscore(cls.__name__)
    UFLType._ufl_all_handler_names_.add(cls._ufl_handler_name_)

    # Append space for counting object creation and destriction of
    # this this type.
    UFLType._ufl_obj_init_counts_.append(0)
    UFLType._ufl_obj_del_counts_.append(0)


def ufl_type(is_abstract=False,
             is_terminal=None,
             is_scalar=False,
             is_index_free=False,
             is_shaping=False,
             is_literal=False,
             is_terminal_modifier=False,
             is_in_reference_frame=False,
             is_restriction=False,
             is_evaluation=False,
             is_differential=None,
             use_default_hash=True,
             num_ops=None,
             inherit_shape_from_operand=None,
             inherit_indices_from_operand=None,
             wraps_type=None,
             unop=None,
             binop=None,
             rbinop=None):
    """This decorator is to be applied to every subclass in the UFL ``Expr`` and ``BaseForm`` hierarchy.

    This decorator contains a number of checks that are
    intended to enforce uniform behaviour across UFL types.

    The rationale behind the checks and the meaning of the
    optional arguments should be sufficiently documented
    in the source code below.
    """

    def _ufl_type_decorator_(cls):

        # Update attributes for UFLType instances (BaseForm and Expr objects)
        update_ufl_type_attributes(cls)
        if not issubclass(cls, core.expr.Expr):
            # Don't need anything else for non Expr subclasses
            return cls

        # is_scalar implies is_index_freeg
        if is_scalar:
            _is_index_free = True
        else:
            _is_index_free = is_index_free

        # Store type traits
        cls._ufl_class_ = cls
        set_trait(cls, "is_abstract", is_abstract, inherit=False)

        set_trait(cls, "is_terminal", is_terminal, inherit=True)
        set_trait(cls, "is_literal", is_literal, inherit=True)
        set_trait(cls, "is_terminal_modifier", is_terminal_modifier,
                  inherit=True)
        set_trait(cls, "is_shaping", is_shaping, inherit=True)
        set_trait(cls, "is_in_reference_frame", is_in_reference_frame,
                  inherit=True)
        set_trait(cls, "is_restriction", is_restriction, inherit=True)
        set_trait(cls, "is_evaluation", is_evaluation, inherit=True)
        set_trait(cls, "is_differential", is_differential, inherit=True)

        set_trait(cls, "is_scalar", is_scalar, inherit=True)
        set_trait(cls, "is_index_free", _is_index_free, inherit=True)

        # Number of operands can often be determined automatically
        _num_ops = determine_num_ops(cls, num_ops, unop, binop, rbinop)
        set_trait(cls, "num_ops", _num_ops)

        # Attach builtin type wrappers to Expr
        """# These are currently handled in the as_ufl implementation in constantvalue.py
        if wraps_type is not None:
            if not isinstance(wraps_type, type):
                msg = "Expecting a type, not a {0.__name__} for the wraps_type argument in definition of {1.__name__}."
                raise TypeError(msg.format(type(wraps_type), cls))

            def _ufl_from_type_(value):
                return cls(value)
            from_type_name = "_ufl_from_{0}_".format(wraps_type.__name__)
            setattr(Expr, from_type_name, staticmethod(_ufl_from_type_))
        """

        # Attach special function to Expr.
        # Avoids the circular dependency problem of making
        # Expr.__foo__ return a Foo that is a subclass of Expr.
        """# These are currently attached in exproperators.py
        if unop:
            def _ufl_expr_unop_(self):
                return cls(self)
            setattr(Expr, unop, _ufl_expr_unop_)
        if binop:
            def _ufl_expr_binop_(self, other):
                try:
                    other = Expr._ufl_coerce_(other)
                except:
                    return NotImplemented
                return cls(self, other)
            setattr(Expr, binop, _ufl_expr_binop_)
        if rbinop:
            def _ufl_expr_rbinop_(self, other):
                try:
                    other = Expr._ufl_coerce_(other)
                except:
                    return NotImplemented
                return cls(other, self)
            setattr(Expr, rbinop, _ufl_expr_rbinop_)
        """

        # Make sure every non-abstract class has its own __hash__ and
        # __eq__.  Python 3 will set __hash__ to None if cls has
        # __eq__, but we've implemented it in a separate function and
        # want to inherit/use that for all types. Allow overriding by
        # setting use_default_hash=False.
        if use_default_hash:
            cls.__hash__ = compute_expr_hash

        # NB! This function conditionally adds some methods to the
        # class!  This approach significantly reduces the amount of
        # small functions to implement across all the types but of
        # course it's a bit more opaque.
        attach_implementations_of_indexing_interface(cls,
                                                     inherit_shape_from_operand,
                                                     inherit_indices_from_operand)

        # Update Expr
        update_global_expr_attributes(cls)

        # Apply a range of consistency checks to detect bugs in type
        # implementations that Python doesn't check for us, including
        # some checks that a static language compiler would do for us
        check_abstract_trait_consistency(cls)
        check_has_slots(cls)
        check_is_terminal_consistency(cls)
        check_implements_required_methods(cls)
        check_implements_required_properties(cls)
        check_type_traits_consistency(cls)

        return cls

    return _ufl_type_decorator_


class UFLType(type):
    """Base class for all UFL types.

    Equip UFL types with some ufl specific properties.
    """

    # A global counter of the number of typecodes assigned.
    _ufl_num_typecodes_ = 0

    # Set the handler name for UFLType
    _ufl_handler_name_ = "ufl_type"

    # A global array of all Expr and BaseForm subclasses, indexed by typecode
    _ufl_all_classes_ = []

    # A global set of all handler names added
    _ufl_all_handler_names_ = set()

    # A global array of the number of initialized objects for each
    # typecode
    _ufl_obj_init_counts_ = []

    # A global array of the number of deleted objects for each
    # typecode
    _ufl_obj_del_counts_ = []

    # Type trait: If the type is abstract.  An abstract class cannot
    # be instantiated and does not need all properties specified.
    _ufl_is_abstract_ = True

    # Type trait: If the type is terminal.
    _ufl_is_terminal_ = None
