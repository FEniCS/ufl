**********
Algorithms
**********

Algorithms to work with UFL forms and expressions can be found in the
submodule ``ufl.algorithms``.  You can import all of them with
the line::

  from ufl.algorithms import *

This chapter gives an overview of (most of) the implemented algorithms.
The intended audience is primarily developers, but advanced users may
find information here useful for debugging.

While domain specific languages introduce notation to express particular
ideas more easily, which can reduce the probability of bugs in user code,
they also add yet another layer of abstraction which can make debugging
more difficult when the need arises.  Many of the utilities described
here can be useful in that regard.


Formatting expressions
======================

Expressions can be formatted in various ways for inspection, which is
particularly useful for debugging.  We use the following as an example
form for the formatting sections below::

  element = FiniteElement("CG", triangle, 1)
  v = TestFunction(element)
  u = TrialFunction(element)
  c = Coefficient(element)
  f = Coefficient(element)
  a = c*u*v*dx + f*v*ds


str
---
Compact human readable pretty printing.  Useful in interactive Python
sessions.  Example output of ``str(a)``::

  TODO

repr
----
Accurate description of expression, with the property that
``eval(repr(a)) == a``.  Useful to see which representation types
occur in an expression, especially if ``str(a)`` is ambiguous.
Example output of ``repr(a)``::

  TODO


Tree formatting
---------------

Ascii tree formatting, useful to inspect the tree structure of
an expression in interactive Python sessions.  Example output of
``tree_format(a)``::

  TODO


Inspecting and manipulating the expression tree
===============================================

This subsection is mostly for form compiler developers and technically
interested users.

TODO: More details about traversal and transformation algorithms for
developers.

Traversing expressions
----------------------

``iter\_expressions``
^^^^^^^^^^^^^^^^^^^^^

Example usage::

  q = f*v
  r = g*v
  s = u*v
  a = q*dx(0) + r*dx(1) + s*ds(0)
  for e in iter_expressions(a):
      print str(e)

``post\_traversal``
^^^^^^^^^^^^^^^^^^^

TODO: traversal.py

``pre\_traversal``
^^^^^^^^^^^^^^^^^^

TODO: traversal.py


``walk``
^^^^^^^^

TODO: traversal.py


``traverse\_terminals``
^^^^^^^^^^^^^^^^^^^^^^^

TODO: traversal.py


Extracting information
----------------------

TODO: analysis.py


Transforming expressions
------------------------

So far the algorithms presented has been about inspecting expressions
in various ways. Some recurring patterns occur when writing algorithms
to modify expressions, either to apply mathematical transformations or
to change their representation. Usually, different expression node types
need different treatment.

To assist in such algorithms, UFL provides the ``Transformer``
class. This implements a variant of the Visitor pattern to enable easy
definition of transformation rules for the types you wish to handle.

Shown here is maybe the simplest transformer possible::

  class Printer(Transformer):
      def __init__(self):
          Transformer.__init__(self)

      def expr(self, o, *operands):
          print "Visiting", str(o), "with operands:"
          print ", ".join(map(str,operands))
          return o

  element = FiniteElement("CG", triangle, 1)
  v = TestFunction(element)
  u = TrialFunction(element)
  a = u*v

  p = Printer()
  p.visit(a)

The call to ``visit`` will traverse ``a`` and call
``Printer.expr`` on all expression nodes in post--order, with the
argument ``operands`` holding the return values from visits to the
operands of ``o``. The output is::

  TODO

Implementing ``expr`` above provides a default handler for any
expression node type. For each subclass of ``Expr`` you can
define a handler function to override the default by using the name
of the type in underscore notation, e.g. ``vector\_constant``
for ``VectorConstant``.  The constructor of ``Transformer``
and implementation of ``Transformer.visit`` handles the mapping
from type to handler function automatically.

Here is a simple example to show how to override default behaviour::

  class CoefficientReplacer(Transformer):
      def __init__(self):
          Transformer.__init__(self)

      expr = Transformer.reuse_if_possible
      terminal = Transformer.always_reuse

      def coefficient(self, o):
          return FloatValue(3.14)

  element = FiniteElement("CG", triangle, 1)
  v = TestFunction(element)
  f = Coefficient(element)
  a = f*v

  r = CoefficientReplacer()
  b = r.visit(a)
  print b

The output of this code is the transformed expression ``b ==
3.14*v``.  This code also demonstrates how to reuse existing handlers.
The handler ``Transformer.reuse\_if\_possible`` will return the
input object if the operands have not changed, and otherwise reconstruct
a new instance of the same type but with the new transformed operands.
The handler ``Transformer.always\_reuse`` always reuses the instance
without recursing into its children, usually applied to terminals.
To set these defaults with less code, inherit ``ReuseTransformer``
instead of ``Transformer``. This ensures that the parts of the
expression tree that are not changed by the transformation algorithms
always reuse the same instances.

We have already mentioned the difference between pre--traversal
and post--traversal, and some times you need to combine the
two. ``Transformer`` makes this easy by checking the number of
arguments to your handler functions to see if they take transformed
operands as input or not.  If a handler function does not take more
than a single argument in addition to self, its children are not visited
automatically, and the handler function must call ``visit`` on its
operands itself.

Here is an example of mixing pre- and post-traversal::

  class Traverser(ReuseTransformer):
      def __init__(self):
          ReuseTransformer.__init__(self)

      def sum(self, o):
          operands = o.operands()
          newoperands = []
          for e in operands:
              newoperands.append( self.visit(e) )
          return sum(newoperands)

  element = FiniteElement("CG", triangle, 1)
  f = Coefficient(element)
  g = Coefficient(element)
  h = Coefficient(element)
  a = f+g+h

  r = Traverser()
  b = r.visit(a)
  print b

This code inherits the ``ReuseTransformer`` like explained above,
so the default behaviour is to recurse into children first and then call
``Transformer.reuse\_if\_possible`` to reuse or reconstruct each
expression node.  Since ``sum`` only takes ``self`` and the
expression node instance ``o`` as arguments, its children are not
visited automatically, and ``sum`` calls on ``self.visit``
to do this explicitly.


Automatic differentiation implementation
========================================

This subsection is mostly for form compiler developers and technically
interested users.

TODO: More details about AD algorithms for developers.


Forward mode
------------

TODO: forward\_ad.py


Reverse mode
------------

TODO: reverse\_ad.py

Mixed derivatives
-----------------

TODO: ad.py


Computational graphs
====================

This section is for form compiler developers and is probably of no
interest to end-users.

An expression tree can be seen as a directed acyclic graph (DAG).
To aid in the implementation of form compilers, UFL includes tools to
build a linearized\footnote{Linearized as in a linear datastructure,
do not confuse this with automatic differentiation.} computational graph
from the abstract expression tree.

A graph can be partitioned into subgraphs based on dependencies of
subexpressions, such that a quadrature based compiler can easily place
subexpressions inside the right sets of loops.

% TODO: Finish and test this before writing about it :)
%The vertices of a graph can be reordered to improve the efficiency
%of the generated code, an operation usually called operation scheduling.

The computational graph
-----------------------

TODO: finish graph.py:

  TODO

Consider the expression:

.. math::

  f = (a + b) * (c + d)

where a, b, c, d are arbitrary scalar expressions.
The *expression tree* for f looks like this::

  TODO: Make figures.
   a   b  c  d
   \  /    \  /
    +      +
      \    /
        *

In UFL f is represented like this expression tree.  If a,b,c,d are all
distinct Coefficient instances, the UFL representation will look like this::

  Coefficient   Coefficient  Coefficient  Coefficient
  \  /    \  /
  Sum      Sum
    \    /
      Product

If we instead have the expression

.. math::

  f = (a + b) * (a - b)

the tree will in fact look like this, with the functions a and b only
represented once::

  Coefficient   Coefficient
  |         \       /       |
  |          Sum        Product -- IntValue(-1)
  |             |            |
  |           Product   |
  |             |           |
  |---------- Sum ------|

The expression tree is a directed acyclic graph (DAG) where the vertices
are Expr instances and each edge represents a direct dependency between
two vertices, i.e. that one vertex is among the operands of another.
A graph can also be represented in a linearized data structure, consisting
of an array of vertices and an array of edges. This representation is
convenient for many algorithms. An example to illustrate this graph
representation::

  G = V, E
  V = [a, b, a+b, c, d, c+d, (a+b)*(c+d)]
  E = [(6,2), (6,5), (5,3), (5,4), (2,0), (2,1)]

In the following this representation of an expression will be called
the *computational graph*.  To construct this graph from a UFL
expression, simply do::

  G = Graph(expression)
  V, E = G

The Graph class can build some useful data structures for use in
algorithms::

  Vin  = G.Vin()  # Vin[i]  = list of vertex indices j such that there is an edge from V[j] to V[i]
  Vout = G.Vout() # Vout[i] = list of vertex indices j such that there is an edge from V[i] to V[j]
  Ein  = G.Ein()  # Ein[i]  = list of edge indices j such that E[j] is an edge to V[i], e.g. E[j][1] == i
  Eout = G.Eout() # Eout[i] = list of edge indices j such that E[j] is an edge from V[i], e.g. E[j][0] == i

The ordering of the vertices in the graph can in principle be arbitrary,
but here they are ordered such that

.. math::

   v_i \prec v_j, \quad \forall j > i,

where :math:`a \prec b` means that :math:a does not depend on :math:b
directly or indirectly.

Another property of the computational graph built by UFL is that no
identical expression is assigned to more than one vertex. This is
achieved efficiently by inserting expressions in a dict (a hash map)
during graph building.

In principle, correct code can be generated for an expression from its
computational graph simply by iterating over the vertices and generating
code for each one separately. However, we can do better than that.


Partitioning the graph
----------------------

To help generate better code efficiently, we can partition vertices by
their dependencies, which allows us to, e.g., place expressions outside
the quadrature loop if they don't depend (directly or indirectly) on
the spatial coordinates. This is done simply by::

  P = partition(G) # TODO
