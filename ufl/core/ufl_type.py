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

import abc
import typing

from ufl.utils.formatting import camel2underscore


class UFLObject(abc.ABC):
    """A UFL Object."""

    _ufl_is_terminal_: bool

    @abstractmethod
    def _ufl_hash_data_(self) -> typing.Hashable:
        """Return hashable data that uniquely defines this object."""
        ...

    @abc.abstractmethod
    def __str__(self) -> str:
        """Return a human-readable string representation of the object."""
        ...

    @abc.abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of the object."""
        ...

    def __hash__(self) -> int:
        """Hash the object."""
        return hash(self._ufl_hash_data_())

    def __eq__(self, other):
        """Check if two objects are equal."""
        return type(self) is type(other) and self._ufl_hash_data_() == other._ufl_hash_data_()

    def __ne__(self, other):
        """Check inequality."""
        return not self.__eq__(other)


def ufl_type():
    """Decorator to apply to every subclass in the UFL ``Expr`` and ``BaseForm`` hierarchy."""

    def _ufl_type_decorator_(cls):
        """UFL type decorator."""
        # Update attributes for UFLType instances (BaseForm and Expr objects)
        # Determine integer typecode by incrementally counting all types
        cls._ufl_typecode_ = UFLRegistry().number_registered_classes
        UFLRegistry().register_class(cls)

        assert UFLRegistry().number_registered_classes == len(UFLRegistry().all_classes)

        # Determine handler name by a mapping from "TypeName" to "type_name"
        cls._ufl_handler_name_ = camel2underscore(cls.__name__)

        return cls

    return _ufl_type_decorator_


class UFLRegistry:
    """Maintains global informations of the registered types."""

    _instance: UFLRegistry | None = None
    _all_classes: list[type]

    # TODO: profiling should be maintained in an own profiling class/registry
    _obj_tracking: dict[type, tuple[int, int]]

    def __new__(cls) -> UFLRegistry:
        """Create singleton UFLRegistry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._all_classes = []
            cls._instance._obj_tracking = {}
        return cls._instance

    @property
    def all_classes(self) -> list[type]:  # list[UFLType]
        """Return list of all Expr and BaseForm subclasses, indexed by typecode."""
        return self._all_classes

    def register_class(self, c: type) -> None:
        """Register an UFLType with the registry."""
        assert c not in self.all_classes
        self._all_classes.append(c)

    @property
    def number_registered_classes(self) -> int:
        """Return number of registered classes."""
        return len(self._all_classes)

    def register_object_creation(self, c: type) -> None:
        """Profiling."""
        data = self._obj_tracking.get(c, (0, 0))
        self._obj_tracking[c] = (data[0] + 1, data[1])

    def register_object_destruction(self, c: type) -> None:
        """Profiling."""
        data = self._obj_tracking.get(c, (0, 0))
        self._obj_tracking[c] = (data[0], data[1] - 1)

    def reset_object_tracking(self) -> None:
        """Profiling."""
        self._obj_tracking = {}

    @property
    def object_tracking(self) -> dict[type, tuple[int, int]]:
        """Profiling."""
        return self._obj_tracking


class UFLType:
    """Base class for all UFL types.

    Equip UFL types with some ufl specific properties.
    """

    # TODO: can we move this assignment into __new__?
    _ufl_typecode_: int

    __slots__: tuple[str, ...] = ()

    # TODO: ufl_handler_name type name -> remove
    _ufl_handler_name_: str = "ufl_type"

    # TODO: _ufl_is_terminal iff. is Cofunction or Terminal -> remove
    _ufl_is_terminal_: bool = False

    # TODO:_ufl_is_literal_ iff. is Zero, ComplexValue, FloatValue or IntValue -> remove
    _ufl_is_literal_: bool = False
