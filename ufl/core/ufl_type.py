"""UFL type."""
# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Massimiliano Leoni, 2016
# Modified by Matthew Scroggs, 2023

import abc
import typing

from ufl.utils.formatting import camel2underscore


class UFLObject(abc.ABC):
    """A UFL Object."""

    @abc.abstractmethod
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


def ufl_type(cls: type | None = None, /, *, handler_name: str | None = None) -> typing.Any:
    """Register a class as a distinct UFL type.

    Registration assigns the class its own typecode and handler name. UFL
    algorithms use these to tell types apart: in dispatch, sorting, hashing and
    form signatures.

    Registration is opt-in. A subclass that is not registered shares the
    typecode and handler name of its closest registered base, so UFL treats it
    as that type. This is intended for subclasses that only add implementation,
    such as a downstream ``Function`` subclass of :class:`~ufl.Coefficient`.

    Use as ``@ufl_type``, or as ``@ufl_type(handler_name=...)`` to choose a
    handler name other than the snake case class name.

    Args:
        cls: The class to register.
        handler_name: The handler name. Defaults to the class name in snake case.
    """

    def register(cls: type) -> type:
        cls._ufl_typecode_ = registry.register(cls)  # type: ignore[attr-defined]
        cls._ufl_handler_name_ = handler_name or camel2underscore(cls.__name__)  # type: ignore[attr-defined]
        return cls

    return register if cls is None else register(cls)


def ufl_handler_names(cls: type) -> typing.Iterator[str]:
    """Yield the handler names to try for a UFL type, most specific first.

    These are the handler names of the registered types in the MRO of ``cls``,
    followed by the ``ufl_type`` fallback. Classes that are not registered, such
    as mixins and implementation subclasses, are skipped.
    """
    for c in cls.__mro__:
        name = c.__dict__.get("_ufl_handler_name_")
        if name is not None:
            yield name


class UFLRegistry:
    """The registered UFL types, indexed by typecode."""

    def __init__(self) -> None:
        """Initialise an empty registry."""
        self._classes: tuple[type, ...] = ()

    @property
    def classes(self) -> tuple[type, ...]:
        """Return the registered types, indexed by typecode."""
        return self._classes

    def __len__(self) -> int:
        """Return the number of registered types."""
        return len(self._classes)

    def register(self, cls: type) -> int:
        """Register a UFL type and return its typecode."""
        if cls in self._classes:
            raise ValueError(f"{cls.__name__} is already registered as a UFL type.")
        self._classes += (cls,)
        return len(self._classes) - 1


registry = UFLRegistry()


class UFLType:
    """Base class for all UFL types.

    Equip UFL types with some ufl specific properties.
    """

    # Set for each registered type by ufl_type.
    _ufl_typecode_: int

    __slots__: tuple[str, ...] = ()

    # Handler name used for dispatch, set for each registered type by
    # ufl_type. This one is the fallback handler for all UFL types.
    _ufl_handler_name_: str = "ufl_type"

    # Whether instances are leaves of the DAG they appear in: expression
    # terminals, and forms such as a Form inside a FormSum.
    _ufl_is_terminal_: bool = False
