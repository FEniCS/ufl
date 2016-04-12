**********
Algorithms
**********

Algorithms to work with UFL forms and expressions can be found in the
submodule ``ufl.algorithms``.  You can import all of them with
the line

::

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
Compact, human readable pretty printing.  Useful in interactive Python
sessions.  Example output of ``str(a)``::

  { v_0 * v_1 * w_0 } * dx(<Mesh #-1 with coordinates parameterized by <Lagrange vector element of degree 1 on a triangle: 2 x <CG1 on a triangle>>>[everywhere], {})
  +  { v_0 * w_1 } * ds(<Mesh #-1 with coordinates parameterized by <Lagrange vector element of degree 1 on a triangle: 2 x <CG1 on a triangle>>>[everywhere], {})

repr
----
Accurate description of an expression, with the property that
``eval(repr(a)) == a``.  Useful to see which representation types
occur in an expression, especially if ``str(a)`` is ambiguous.
Example output of ``repr(a)``::

  Form([Integral(Product(Argument(FunctionSpace(Mesh(VectorElement('Lagrange', triangle, 1, dim=2), -1), FiniteElement('Lagrange', triangle, 1)), 0, None), Product(Argument(FunctionSpace(Mesh(VectorElement('Lagrange', triangle, 1, dim=2), -1), FiniteElement('Lagrange', triangle, 1)), 1, None), Coefficient(FunctionSpace(Mesh(VectorElement('Lagrange', triangle, 1, dim=2), -1), FiniteElement('Lagrange', triangle, 1)), 0))), 'cell', Mesh(VectorElement('Lagrange', triangle, 1, dim=2), -1), 'everywhere', {}, None), Integral(Product(Argument(FunctionSpace(Mesh(VectorElement('Lagrange', triangle, 1, dim=2), -1), FiniteElement('Lagrange', triangle, 1)), 0, None), Coefficient(FunctionSpace(Mesh(VectorElement('Lagrange', triangle, 1, dim=2), -1), FiniteElement('Lagrange', triangle, 1)), 1)), 'exterior_facet', Mesh(VectorElement('Lagrange', triangle, 1, dim=2), -1), 'everywhere', {}, None)])


Tree formatting
---------------

ASCII tree formatting, useful to inspect the tree structure of
an expression in interactive Python sessions.  Example output of
``tree_format(a)``::

  Form:
    Integral:
      integral type: cell
      subdomain id: everywhere
      integrand:
        Product
        (
          Argument(FunctionSpace(Mesh(VectorElement('Lagrange', triangle, 1, dim=2), -1), FiniteElement('Lagrange', triangle, 1)), 0, None)
          Product
          (
            Argument(FunctionSpace(Mesh(VectorElement('Lagrange', triangle, 1, dim=2), -1), FiniteElement('Lagrange', triangle, 1)), 1, None)
            Coefficient(FunctionSpace(Mesh(VectorElement('Lagrange', triangle, 1, dim=2), -1), FiniteElement('Lagrange', triangle, 1)), 0)
          )
        )
  Integral:
    integral type: exterior_facet
    subdomain id: everywhere
    integrand:
      Product
      (
        Argument(FunctionSpace(Mesh(VectorElement('Lagrange', triangle, 1, dim=2), -1), FiniteElement('Lagrange', triangle, 1)), 0, None)
        Coefficient(FunctionSpace(Mesh(VectorElement('Lagrange', triangle, 1, dim=2), -1), FiniteElement('Lagrange', triangle, 1)), 1)
      )


Inspecting and manipulating the expression tree
===============================================

This subsection is mostly for form compiler developers and technically
interested users.

Traversing expressions
----------------------

``iter_expressions``
^^^^^^^^^^^^^^^^^^^^^

Example usage::

  for e in iter_expressions(a):
      print str(e)

outputs::

  v_0 * v_1 * w_0
  v_0 * w_1

..
    ``post_traversal``
    ^^^^^^^^^^^^^^^^^^^

..
    TODO: traversal.py

..
    ``pre_traversal``
    ^^^^^^^^^^^^^^^^^^

..
    TODO: traversal.py


..
    ``walk``
    ^^^^^^^^

..
    TODO: traversal.py


..
    ``traverse_terminals``
    ^^^^^^^^^^^^^^^^^^^^^^^

..
    TODO: traversal.py


..
    Extracting information
    ----------------------

..
    TODO: analysis.py


Transforming expressions
------------------------

So far we presented algorithms meant to inspect expressions
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

  Visiting v_0 * v_1 with operands:
  v_0, v_1

:math:`(v^0_h)(v^1_h)`

Implementing ``expr`` above provides a default handler for any
expression node type. For each subclass of ``Expr`` you can
define a handler function to override the default by using the name
of the type in underscore notation, e.g. ``vector_constant``
for ``VectorConstant``.  The constructor of ``Transformer``
and implementation of ``Transformer.visit`` handles the mapping
from type to handler function automatically.

Here is a simple example to show how to override default behaviour::

  from ufl.classes import *
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

which outputs

::

  3.14 * v_0

The output of this code is the transformed expression ``b ==
3.14*v``.  This code also demonstrates how to reuse existing handlers.
The handler ``Transformer.reuse_if_possible`` will return the
input object if the operands have not changed, and otherwise reconstruct
a new instance of the same type but with the new transformed operands.
The handler ``Transformer.always_reuse`` always reuses the instance
without recursing into its children, usually applied to terminals.
To set these defaults with less code, inherit ``ReuseTransformer``
instead of ``Transformer``. This ensures that the parts of the
expression tree that are not changed by the transformation algorithms
will always reuse the same instances.

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

This code inherits the ``ReuseTransformer`` as explained above,
so the default behaviour is to recurse into children first and then call
``Transformer.reuse_if_possible`` to reuse or reconstruct each
expression node.  Since ``sum`` only takes ``self`` and the
expression node instance ``o`` as arguments, its children are not
visited automatically, and ``sum`` explicitly calls ``self.visit``
to do this.


Automatic differentiation implementation
========================================

This subsection is mostly for form compiler developers and technically
interested users.

First of all, we give a brief explanation of the algorithm.
Recall that a ``Coefficient`` represents a
sum of unknown coefficients multiplied with unknown
basis functions in some finite element space.

.. math::

   w(x) = \sum_k w_k \phi_k(x)

Also recall that an ``Argument`` represents any (unknown) basis
function in some finite element space.

.. math::

   v(x) = \phi_k(x), \qquad \phi_k \in V_h .

A form :math:`L(v; w)` implemented in UFL is intended for discretization
like

.. math::

   b_i = L(\phi_i; \sum_k w_k \phi_k), \qquad \forall \phi_i \in V_h .

The Jacobi matrix :math:`A_{ij}` of this vector can be obtained by
differentiation of :math:`b_i` w.r.t. :math:`w_j`, which can be written

.. math::

   A_{ij} = \frac{d b_i}{d w_j} = a(\phi_i, \phi_j; \sum_k w_k \phi_k), \qquad \forall \phi_i \in V_h, \quad \forall \phi_j \in V_h ,

for some form `a`. In UFL, the form `a` can be obtained by
differentiating `L`.  To manage this, we note that as long as the domain
:math:`\Omega` is independent of :math:`w_j`, :math:`\int_\Omega` commutes with :math:`\frac{d}{d
w_j}`, and we can differentiate the integrand expression instead, e.g.,

.. math::

   L(v; w) = \int_\Omega I_c(v; w) \, dx + \int_{\partial\Omega} I_e(v; w) \, ds, \\
      \frac{d}{d w_j} L(v; w) = \int_\Omega \frac{d I_c}{d w_j} \, dx + \int_{\partial\Omega} \frac{d I_e}{d w_j} \, ds.

In addition, we need that

.. math::

   \frac{d w}{d w_j} = \phi_j, \qquad \forall \phi_j \in V_h ,

which in UFL can be represented as

.. math::

   w &= \mathtt{Coefficient(element)}, \\
   v &= \mathtt{Argument(element)}, \\
   \frac{dw}{d w_j} &= v,

since :math:`w` represents the sum and :math:`v` represents any and all
basis functions in :math:`V_h`.

Other operators have well defined derivatives, and by repeatedly applying
the chain rule we can differentiate the integrand automatically.


..
    TODO: More details about AD algorithms for developers.


..
    Forward mode
    ------------

..
    TODO: forward_ad.py


..
    Reverse mode
    ------------

..
    TODO: reverse_ad.py

..
    Mixed derivatives
    -----------------

..
    TODO: ad.py


Computational graphs
====================

This section is for form compiler developers and is probably of no
interest to end-users.

An expression tree can be seen as a directed acyclic graph (DAG).
To aid in the implementation of form compilers, UFL includes tools to
build a linearized [#]_ computational graph from the abstract expression tree.

A graph can be partitioned into subgraphs based on dependencies of
subexpressions, such that a quadrature based compiler can easily place
subexpressions inside the right sets of loops.

.. [#] Linearized as in a linear datastructure,
   do not confuse this with automatic differentiation.

..
    TODO: Finish and test this before writing about it :)
    The vertices of a graph can be reordered to improve the efficiency
    of the generated code, an operation usually called operation scheduling.

The computational graph
-----------------------
..
    TODO: finish graph.py:

Consider the expression

.. math::

  f = (a + b) * (c + d)

where a, b, c, d are arbitrary scalar expressions.
The *expression tree* for f looks like this::

   a   b   c   d
   \   /   \   /
     +       +
      \     /
         *

In UFL f is represented like this expression tree.  If a, b, c, d are all
distinct Coefficient instances, the UFL representation will look like this::

  Coefficient Coefficient Coefficient Coefficient
            \     /             \     /
              Sum                 Sum
                 \               /
                  --- Product ---

If we instead have the expression

.. math::

  f = (a + b) * (a - b)

the tree will in fact look like this, with the functions a and b only
represented once::

  Coefficient     Coefficient
     |       \   /       |
     |        Sum      Product -- IntValue(-1)
     |         |         |
     |       Product     |
     |         |         |
     |------- Sum -------|

The expression tree is a directed acyclic graph (DAG) where the vertices
are Expr instances and each edge represents a direct dependency between
two vertices, i.e. that one vertex is among the operands of another.
A graph can also be represented in a linearized data structure, consisting
of an array of vertices and an array of edges. This representation is
convenient for many algorithms. An example to illustrate this graph
representation follows::

  G = V, E
  V = [a, b, a+b, c, d, c+d, (a+b)*(c+d)]
  E = [(6,2), (6,5), (5,3), (5,4), (2,0), (2,1)]

In the following, this representation of an expression will be called
the *computational graph*.  To construct this graph from a UFL
expression, simply do

::

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

where :math:`a \prec b` means that :math:`a` does not depend on :math:`b`
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
the spatial coordinates. This is done simply by

..
    TODO

::

  P = partition(G)
