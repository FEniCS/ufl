*******************************
Internal representation details
*******************************
\label{chapter:representation}

This chapter explains how UFL forms and expressions are represented
in detail. Most operations are mirrored by a representation class,
e.g., \texttt{Sum} and \texttt{Product}, all which are subclasses
of \texttt{Expr}.  You can import all of them from the submodule
\texttt{ufl.classes} by::

  from ufl.classes import *

TODO: Automate the construction of class hierarchy figures using ptex2tex.

Structure of a form
===================

TODO: Add class relations figure with Form, Integral, Expr, Terminal,
Operator.

Each \ttt{Form} owns multiple \ttt{Integral} instances, each associated
with a different \ttt{Measure}.  An \ttt{Integral} owns a \ttt{Measure}
and an \ttt{Expr}, which represents the integrand expression. The
\ttt{Expr} is the base class of all expressions.  It has two direct
subclasses \ttt{Terminal} and \ttt{Operator}.

Subclasses of \ttt{Terminal} represent atomic quantities which
terminate the expression tree, e.g. they have no subexpressions.
Subclasses of \ttt{Operator} represent operations on one or more
other expressions, which may usually be \ttt{Expr} subclasses of
arbitrary type. Different \ttt{Operator}s may have restrictions
on some properties of their arguments.

All the types mentioned here are conceptually immutable, i.e. they
should never be modified over the course of their entire lifetime. When a
modified expression, measure, integral, or form is needed, a new instance
must be created, possibly sharing some data with the old one. Since the
shared data is also immutable, sharing can cause no problems.

General properties of expressions
=================================

Any UFL expression has certain properties, defined by functions that
every \ttt{Expr} subclass must implement. In the following, \ttt{u}
represents an arbitrary UFL expression, i.e. an instance of an
arbitrary \ttt{Expr} subclass.

``operands``
------------

``u.operands()`` returns a tuple with all the operands of u, which should
all be \ttt{Expr} instances.

``reconstruct``
---------------

\ttt{u.reconstruct(operands)} returns a new \ttt{Expr} instance
representing the same operation as \ttt{u} but with other
operands. Terminal objects may simply return \ttt{self} since all
\ttt{Expr} instance are immutable.  An important invariant is that
\ttt{u.reconstruct(u.operands()) == u}.

``cell``
--------

\ttt{u.cell()} returns the first \ttt{Cell} instance found in \ttt{u}. It
is currently assumed in UFL that no two different cells are used in
a single form. Not all expression define a cell, in which case this
returns \ttt{None} and \ttt{u} is spatially constant.  Note that this
property is used in some algorithms.


``shape``
---------

\ttt{u.shape()} returns a tuple of integers, which is the tensor shape
of \ttt{u}.


``free\_indices``
-----------------

\ttt{u.free\_indices()} returns a tuple of \ttt{Index} objects, which
are the unassigned, free indices of \ttt{u}.


``index\_dimensions``
---------------------

\ttt{u.index\_dimensions()} returns a \ttt{dict} mapping from each
\ttt{Index} instance in \ttt{u.free\_indices()} to the integer dimension
of the value space each index can range over.


``str(u)``
----------

\ttt{str(u)} returns a human-readable string representation of \ttt{u}.


''repr(u)''
-----------

\ttt{repr(u)} returns a Python string representation of \ttt{u}, such
that \ttt{eval(repr(u)) == u} holds in Python.


``hash(u)``
-----------

\ttt{hash(u)} returns a hash code for \ttt{u}, which is used extensively
(indirectly) in algorithms whenever \ttt{u} is placed in a Python
\ttt{dict} or \ttt{set}.


``u == v``
----------

\ttt{u == v} returns true if and only if \ttt{u} and \ttt{v} represents
the same expression in the exact same way.  This is used extensively
(indirectly) in algorithms whenever \ttt{u} is placed in a Python
\ttt{dict} or \ttt{set}.


About other relational operators
--------------------------------

In general, UFL expressions are not possible to fully evaluate since the
cell and the values of form arguments are not available. Implementing
relational operators for immediate evaluation is therefore impossible.

Overloading relational operators as a part of the form language is not
possible either, since it interferes with the correct use of container
types in Python like \ttt{dict} or \ttt{set}.


Elements
========

All finite element classes have a common base class
\texttt{FiniteElementBase}. The class hierarchy looks like this:

TODO: Class figure.

TODO: Describe all FiniteElementBase subclasses here.


Terminals
=========

All \texttt{Terminal} subclasses have some non-\texttt{Expr} data attached
to them. \texttt{ScalarValue} has a Python scalar, \texttt{Coefficient}
has a \texttt{FiniteElement}, etc.

Therefore, a unified implementation of \texttt{reconstruct} is
not possible, but since all \texttt{Expr} instances are immutable,
\texttt{reconstruct} for terminals can simply return self. This feature
and the immutability property is used extensively in algorithms.

TODO: Describe all Terminal representation classes here.


Operators
=========

All instances of \texttt{Operator} subclasses are fully specified
by their type plus the tuple of \texttt{Expr} instances that are
the operands. Their constructors should take these operands as the
positional arguments, and only that. This way, a unified implementation
of \texttt{reconstruct} is possible, by simply calling the constructor
with new operands. This feature is used extensively in algorithms.

TODO: Describe all Operator representation classes here.


Extending UFL
=============

Adding new types to the UFL class hierarchy must be done with care. If
you can get away with implementing a new operator as a combination of
existing ones, that is the easiest route. The reason is that only some
of the properties of an operator is represented by the \texttt{Expr}
subclass. Other properties are part of the various algorithms in
UFL. One example is derivatives, which are defined in the differentiation
algorithm, and how to render a type to the \LaTeX{} or dot formats. These
properties could be merged into the class hierarchy, but other properties
like how to map a UFL type to some \ffc{} or \sfc{} or \dolfin{} type
can not be part of UFL. So before adding a new class, consider that doing
so may require changes in multiple algorithms and even other projects.

TODO: More issues to consider when adding stuff to ufl.

