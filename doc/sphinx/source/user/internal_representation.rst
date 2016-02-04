*******************************
Internal representation details
*******************************

FIXME: This chapter is very much outdated.
Most of the concepts are still the same but
a lot of the details are different.


This chapter explains how UFL forms and expressions are represented
in detail. Most operations are mirrored by a representation class,
e.g., ``Sum`` and ``Product``, which are subclasses
of ``Expr``.  You can import all of them from the submodule
``ufl.classes`` by::

  from ufl.classes import *

Structure of a form
===================

Each ``Form`` owns multiple ``Integral`` instances, each associated
with a different ``Measure``.  An ``Integral`` owns a ``Measure``
and an ``Expr``, which represents the integrand expression. The
``Expr`` is the base class of all expressions.  It has two direct
subclasses ``Terminal`` and ``Operator``.

Subclasses of ``Terminal`` represent atomic quantities which
terminate the expression tree, e.g. they have no subexpressions.
Subclasses of ``Operator`` represent operations on one or more
other expressions, which may usually be ``Expr`` subclasses of
arbitrary type. Different ``Operator``\ s may have restrictions
on some properties of their arguments.

All the types mentioned here are conceptually immutable, i.e. they
should never be modified over the course of their entire lifetime. When a
modified expression, measure, integral, or form is needed, a new instance
must be created, possibly sharing some data with the old one. Since the
shared data is also immutable, sharing can cause no problems.

General properties of expressions
=================================

Any UFL expression has certain properties, defined by functions that
every ``Expr`` subclass must implement. In the following, ``u``
represents an arbitrary UFL expression, i.e. an instance of an
arbitrary ``Expr`` subclass.

``operands``
------------

``u.operands()`` returns a tuple with all the operands of u, which should
all be ``Expr`` instances.

``reconstruct``
---------------

``u.reconstruct(operands)`` returns a new ``Expr`` instance
representing the same operation as ``u`` but with other
operands. Terminal objects may simply return ``self`` since all
``Expr`` instance are immutable.  An important invariant is that
``u.reconstruct(u.operands()) == u``.

``cell``
--------

``u.cell()`` returns the first ``Cell`` instance found in ``u``. It
is currently assumed in UFL that no two different cells are used in
a single form. Not all expression define a cell, in which case this
returns ``None`` and ``u`` is spatially constant.  Note that this
property is used in some algorithms.


``shape``
---------

``u.shape()`` returns a tuple of integers, which is the tensor shape
of ``u``.


``free_indices``
-----------------

``u.free_indices()`` returns a tuple of ``Index`` objects, which
are the unassigned, free indices of ``u``.


``index_dimensions``
---------------------

``u.index_dimensions()`` returns a ``dict`` mapping from each
``Index`` instance in ``u.free_indices()`` to the integer dimension
of the value space each index can range over.


``str(u)``
----------

``str(u)`` returns a human-readable string representation of ``u``.


``repr(u)``
-----------

``repr(u)`` returns a Python string representation of ``u``, such
that ``eval(repr(u)) == u`` holds in Python.


``hash(u)``
-----------

``hash(u)`` returns a hash code for ``u``, which is used extensively
(indirectly) in algorithms whenever ``u`` is placed in a Python
``dict`` or ``set``.


``u == v``
----------

``u == v`` returns true if and only if ``u`` and ``v`` represents
the same expression in the exact same way.  This is used extensively
(indirectly) in algorithms whenever ``u`` is placed in a Python
``dict`` or ``set``.


About other relational operators
--------------------------------

In general, UFL expressions are not possible to fully evaluate since the
cell and the values of form arguments are not available. Implementing
relational operators for immediate evaluation is therefore impossible.

Overloading relational operators as a part of the form language is not
possible either, since it interferes with the correct use of container
types in Python like ``dict`` or ``set``.


Elements
========

All finite element classes have a common base class
``FiniteElementBase``. The class hierarchy looks like this:

TODO: Class figure.

TODO: Describe all FiniteElementBase subclasses here.


Terminals
=========

All ``Terminal`` subclasses have some non-``Expr`` data attached
to them. ``ScalarValue`` has a Python scalar, ``Coefficient``
has a ``FiniteElement``, etc.

Therefore, a unified implementation of ``reconstruct`` is
not possible, but since all ``Expr`` instances are immutable,
``reconstruct`` for terminals can simply return self. This feature
and the immutability property is used extensively in algorithms.

Operators
=========

All instances of ``Operator`` subclasses are fully specified
by their type plus the tuple of ``Expr`` instances that are
the operands. Their constructors should take these operands as the
positional arguments, and only that. This way, a unified implementation
of ``reconstruct`` is possible, by simply calling the constructor
with new operands. This feature is used extensively in algorithms.

Extending UFL
=============

Adding new types to the UFL class hierarchy must be done with care. If
you can get away with implementing a new operator as a combination of
existing ones, that is the easiest route. The reason is that only some
of the properties of an operator is represented by the ``Expr``
subclass. Other properties are part of the various algorithms in
UFL. One example is derivatives, which are defined in the differentiation
algorithm, and how to render a type to the ``LaTeX`` or dot formats. These
properties could be merged into the class hierarchy, but other properties
like how to map a UFL type to some ``ffc`` or ``dolfin`` type
cannot be part of UFL. So before adding a new class, consider that doing
so may require changes in multiple algorithms and even other projects.
