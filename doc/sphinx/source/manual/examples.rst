*************
Example forms
*************

The following examples illustrate basic usage of the form language
for the definition of a collection of standard multilinear forms. We
assume that ``dx`` has been declared as an integral over the interior of
:math:`\Omega` and that both ``i`` and ``j`` have been declared as a free
``Index``.

The examples presented below can all be found in the subdirectory
``demo/`` of the UFL source tree together with numerous
other examples.

The mass matrix
===============

As a first example, consider the bilinear form corresponding to a
mass matrix,

.. math::

   a(v, u) = \int_{\Omega} v \, u \mathop{dx},

which can be implemented in UFL as follows::

  element = FiniteElement("Lagrange", triangle, 1)

  v = TestFunction(element)
  u = TrialFunction(element)

  a = v*u*dx

This example is implemented in the file ``Mass.ufl`` in the collection
of demonstration forms included with the UFL source distribution.

Poisson equation
================

The bilinear and linear forms form for Poisson's equation,

.. math::

   a(v, u) &= \int_{\Omega} \nabla v \cdot \nabla u \mathop{dx}, \\
   L(v; f) &= \int_{\Omega} v \, f \mathop{dx},

can be implemented as follows::

  element = FiniteElement("Lagrange", triangle, 1)

  v = TestFunction(element)
  u = TrialFunction(element)
  f = Coefficient(element)

  a = dot(grad(v), grad(u))*dx
  L = v*f*dx

Alternatively, index notation can be used to express the scalar product
like this::

  a = Dx(v, i)*Dx(u, i)*dx

or like this::

  a = v.dx(i)*u.dx(i)*dx

This example is implemented in the file ``Poisson.ufl`` in the collection
of demonstration forms included with the UFL source distribution.


Vector-valued Poisson
=====================

The bilinear and linear forms for a system of (independent) Poisson
equations,

.. math::

   a(v, u) &= \int_{\Omega} \nabla v : \nabla u \mathop{dx}, \\
   L(v; f) &= \int_{\Omega} v \cdot f \mathop{dx},

with :math:`v`, :math:`u` and :math:`f` vector-valued can be implemented
as follows::

  element = VectorElement("Lagrange", triangle, 1)

  v = TestFunction(element)
  u = TrialFunction(element)
  f = Coefficient(element)

  a = inner(grad(v), grad(u))*dx
  L = dot(v, f)*dx

Alternatively, index notation may be used like this::

  a = Dx(v[i], j)*Dx(u[i], j)*dx
  L = v[i]*f[i]*dx

or like this::

  a = v[i].dx(j)*u[i].dx(j)*dx
  L = v[i]*f[i]*dx

This example is implemented in the file ``PoissonSystem.ufl`` in
the collection of demonstration forms included with the UFL source
distribution.


The strain-strain term of linear elasticity
===========================================

The strain-strain term of linear elasticity,

.. math::

   a(v, u) = \int_{\Omega} \epsilon(v) : \epsilon(u) \mathop{dx},

where

.. math::

   \epsilon(v) = \frac{1}{2}(\nabla v + (\nabla v)^{\top})

can be implemented as follows::

  element = VectorElement("Lagrange", tetrahedron, 1)

  v = TestFunction(element)
  u = TrialFunction(element)

  def epsilon(v):
      Dv = grad(v)
      return 0.5*(Dv + Dv.T)

  a = inner(epsilon(v), epsilon(u))*dx

Alternatively, index notation can be used to define the form::

  a = 0.25*(Dx(v[j], i) + Dx(v[i], j))* \
           (Dx(u[j], i) + Dx(u[i], j))*dx

or like this::

  a = 0.25*(v[j].dx(i) + v[i].dx(j))* \
           (u[j].dx(i) + u[i].dx(j))*dx

This example is implemented in the file ``Elasticity.ufl`` in the
collection of demonstration forms included with the UFL source
distribution.


The nonlinear term of Navier--Stokes
====================================

The bilinear form for fixed-point iteration on the nonlinear term of
the incompressible Navier--Stokes equations,

.. math::

   a(v, u; w) = \int_{\Omega} (w \cdot \nabla u) \cdot v \mathop{dx},

with :math:`w` the frozen velocity from a previous iteration, can be
implemented as follows::

  element = VectorElement("Lagrange", tetrahedron, 1)

  v = TestFunction(element)
  u = TrialFunction(element)
  w = Coefficient(element)

  a = dot(grad(u)*w, v)*dx

alternatively using index notation like this::

  a = v[i]*w[j]*Dx(u[i], j)*dx

or like this::

  a = v[i]*w[j]*u[i].dx(j)*dx

This example is implemented in the file ``NavierStokes.ufl`` in
the collection of demonstration forms included with the UFL source
distribution.

The heat equation
=================

Discretizing the heat equation,

.. math::

   \dot{u} - \nabla \cdot (c \nabla u) = f,

in time using the :math:`\mathrm{dG}(0)` method (backward Euler), we
obtain the following variational problem for the discrete solution :math:`u_h
= u_h(x, t)`: Find :math:`u_h^n = u_h(\cdot, t_n)` with
:math:`u_h^{n-1} = u_h(\cdot, t_{n-1})` given such that

.. math::

   \frac{1}{k_n} \int_{\Omega} v \, (u_h^n - u_h^{n-1}) \mathop{dx} +
   \int_{\Omega} c \, \nabla v \cdot \nabla u_h^n \mathop{dx} =
   \int_{\Omega} v \, f^n \mathop{dx}

for all test functions :math:`v`, where :math:`k_n = t_n - t_{n-1}`
denotes the time step. In the example below, we implement this
variational problem with piecewise linear test and trial functions,
but other choices are possible (just choose another finite element).

Rewriting the variational problem in the standard form :math:`a(v, u_h)
= L(v)` for all :math:`v`, we obtain the following pair of bilinear and
linear forms:

.. math::

  a(v, u_h^n; c, k) &= \int_{\Omega} v \, u_h^n \mathop{dx} +
  k_n \int_{\Omega} c \, \nabla v \cdot \nabla u_h^n \mathop{dx}, \\
  L(v; u_h^{n-1}, f, k) &= \int_{\Omega} v \, u_h^{n-1} \mathop{dx} + k_n \int_{\Omega} v \, f^n \mathop{dx},

which can be implemented as follows::

  element = FiniteElement("Lagrange", triangle, 1)

  v  = TestFunction(element)  # Test function
  u1 = TrialFunction(element) # Value at t_n
  u0 = Coefficient(element)   # Value at t_n-1
  c  = Coefficient(element)   # Heat conductivity
  f  = Coefficient(element)   # Heat source
  k  = Constant("triangle")   # Time step

  a = v*u1*dx + k*c*dot(grad(v), grad(u1))*dx
  L = v*u0*dx + k*v*f*dx

This example is implemented in the file ``Heat.ufl`` in the collection
of demonstration forms included with the UFL source distribution.


Mixed formulation of Stokes
===========================

To solve Stokes' equations,

.. math::

  - \Delta u + \nabla p &= f, \\
  \nabla \cdot u &= 0,

we write the variational problem in standard form :math:`a(v, u) =
L(v)` for all :math:`v` to obtain the following pair of bilinear and
linear forms:

.. math::

   a((v, q), (u, p)) &= \int_{\Omega} \nabla v : \nabla u - (\nabla \cdot v) \, p +
   q \, (\nabla \cdot u) \mathop{dx}, \\
   L((v, q); f) &= \int_{\Omega} v \cdot f \mathop{dx}.

Using a mixed formulation with Taylor-Hood elements, this can be
implemented as follows::

  cell = triangle
  P2 = VectorElement("Lagrange", cell, 2)
  P1 = FiniteElement("Lagrange", cell, 1)
  TH = P2 * P1

  (v, q) = TestFunctions(TH)
  (u, p) = TrialFunctions(TH)

  f = Coefficient(P2)

  a = (inner(grad(v), grad(u)) - div(v)*p + q*div(u))*dx
  L = dot(v, f)*dx

This example is implemented in the file ``Stokes.ufl`` in the collection
of demonstration forms included with the UFL source distribution.

Mixed formulation of Poisson
============================

We next consider the following formulation of Poisson's equation as a
pair of first order equations for :math:`\sigma \in H(\mathrm{div})`
and :math:`u \in L^2`:

.. math::
   \sigma + \nabla u &= 0, \\
   \nabla \cdot \sigma &= f.

We multiply the two equations by a pair of test functions :math:`\tau` and
:math:`w` and integrate by parts to obtain the following variational
problem: Find :math:`(\sigma, u) \in V = H(\mathrm{div}) \times L^2` such that

.. math::

   a((\tau, w), (\sigma, u)) = L((\tau, w)) \quad \forall \, (\tau, w) \in V,

where

.. math::

   a((\tau, w), (\sigma, u)) &= \int_{\Omega} \tau \cdot \sigma - \nabla \cdot \tau \, u
   + w \nabla \cdot \sigma \mathop{dx},
   \\
   L((\tau, w); f) &= \int_{\Omega} w \cdot f \mathop{dx}.

We may implement the corresponding forms in our form language using
first order BDM H(div)-conforming elements for
:math:`\sigma` and piecewise constant :math:`L^2`-conforming elements for
:math:`u` as follows::

  cell = triangle
  BDM1 = FiniteElement("Brezzi-Douglas-Marini", cell, 1)
  DG0  = FiniteElement("Discontinuous Lagrange", cell, 0)

  element = BDM1 * DG0

  (tau, w) = TestFunctions(element)
  (sigma, u) = TrialFunctions(element)

  f = Coefficient(DG0)

  a = (dot(tau, sigma) - div(tau)*u + w*div(sigma))*dx
  L = w*f*dx

This example is implemented in the file ``MixedPoisson.ufl`` in
the collection of demonstration forms included with the UFL source
distribution.

Poisson equation with DG elements
=================================

We consider again Poisson's equation, but now in an (interior penalty)
discontinuous Galerkin formulation: Find :math:`u \in V = L^2` such that

.. math::

   a(v, u) = L(v) \quad \forall v \in V,

where

.. math::

   a(v, u; h) &= \int_{\Omega} \nabla v \cdot \nabla u \mathop{dx} \\
   &+ \sum_S \int_S
   - \langle \nabla v \rangle \cdot [[ u ]]_n
   - [[ v ]]_n \cdot \langle \nabla u \rangle
   + (\alpha/h) [[ v ]]_n \cdot [[ u ]]_n \mathop{dS} \\
   &+ \int_{\partial\Omega}
   - \nabla v \cdot [[ u ]]_n - [[ v ]]_n \cdot \nabla u
   + (\gamma/h) v u \mathop{ds} \\
   L(v; f, g) &= \int_{\Omega} v f \mathop{dx} + \int_{\partial\Omega} v g \mathop{ds}.

The corresponding finite element variational problem for discontinuous
first order elements may be implemented as follows::

  cell = triangle
  DG1 = FiniteElement("Discontinuous Lagrange", cell, 1)

  v = TestFunction(DG1)
  u = TrialFunction(DG1)

  f = Coefficient(DG1)
  g = Coefficient(DG1)
  h = 2.0*Circumradius(cell)
  alpha = 1
  gamma = 1

  a = dot(grad(v), grad(u))*dx \
    - dot(avg(grad(v)), jump(u))*dS \
    - dot(jump(v), avg(grad(u)))*dS \
    + alpha/h('+')*dot(jump(v), jump(u))*dS \
    - dot(grad(v), jump(u))*ds \
    - dot(jump(v), grad(u))*ds \
    + gamma/h*v*u*ds
  L = v*f*dx + v*g*ds

..
    TODO: set alpha and gamma to proper values

This example is implemented in the file ``PoissonDG.ufl`` in
the collection of demonstration forms included with the UFL source
distribution.

Poisson equation with Dirichlet boundary conditions
===================================================

The bilinear and linear forms for Poisson's equation with Dirichlet
boundary condition,

.. math::

   \int_{\partial\Omega} u \mathop{dx} &= \int_{\partial\Omega} g \mathop{dx},

is implemented in the following. As before, we have::

  element = FiniteElement("Lagrange", triangle, 1)

  v = TestFunction(element)
  u = TrialFunction(element)
  f = Coefficient(element)

Here, we also define a coefficient for the Dirichlet boundary value as::

  g = Coefficient(element)

We then mark degrees of freedom for which we implement the
domain equation::

  transform_op_0 = Subspace(element)

and those on which the Dirichlet condition is enforced::

  transform_op_1 = Subspace(element)

We then project :math:`v`, :math:`u`, and :math:`g` to
appropriate subspaces as::

  v0 = Masked(v, transform_op_0)
  v1 = Masked(v, transform_op_1)
  u0 = Masked(u, transform_op_0)
  u1 = Masked(u, transform_op_1)
  g1 = Masked(g, transform_op_1)

The Poisson equation with Dirichlet boundary condition is
then implemented as::

  a = dot(grad(u0), grad(v0)) * dx + u1 * v1 * ds
  L = f * v0 * dx - dot(grad(g1), grad(v0)) * dx + g * v1 * ds

Note that the bilinear form :math:`a` is symmetrized.

The Quadrature family
=====================

..
    *FIXME: The code examples in this section have been mostly converted
    to UFL syntax, but the quadrature elements need some more updating, as
    well as the text.  In UFL, I think we should define the element order
    and not the number of points for quadrature elements, and let the form
    compiler choose a quadrature rule.  This way the form depends less on
    the cell in use.*

We consider here a nonlinear version of the Poisson's equation to
illustrate the main point of the ``Quadrature`` finite element
family. The strong equation looks as follows:

.. math::

  - \nabla \cdot (1+u^2)\nabla u = f.

The linearised bilinear and linear forms for this equation,

.. math::

   a(v, u; u_0) &= \int_{\Omega} (1+u_{0}^2) \nabla v \cdot \nabla u \mathop{dx}
   + \int_{\Omega} 2u_0 u \nabla v \cdot \nabla u_0 \mathop{dx},
   \\
   L(v; u_0, f)    &= \int_{\Omega} v \, f \mathop{dx}
   - \int_{\Omega} (1+u_{0}^2) \nabla v \cdot \nabla u_0 \mathop{dx},

can be implemented in a single form file as follows::

  element = FiniteElement("Lagrange", triangle, 1)

  v = TestFunction(element)
  u = TrialFunction(element)
  u0 = Coefficient(element)
  f = Coefficient(element)

  a = (1+u0**2)*dot(grad(v), grad(u))*dx + 2*u0*u*dot(grad(v), grad(u0))*dx
  L = v*f*dx - (1+u0**2)*dot(grad(v), grad(u0))*dx

Here, :math:`u_0` represents the solution from the previous Newton-Raphson
iteration.

The above form will be denoted REF1 and serves as our reference
implementation for linear elements. A similar form (REF2) using quadratic
elements will serve as a reference for quadratic elements.

Now, assume that we want to treat the quantities :math:`C = (1 + u_{0}^2)`
and :math:`\sigma_0 = (1+u_{0}^2) \nabla u_0` as given functions (to be
computed elsewhere). Substituting into the bilinear and linear forms, we obtain

.. math::
   a(v, u) &= \int_{\Omega} \text{C} \nabla v \cdot \nabla u \mathop{dx}
   + \int_{\Omega} 2u_0 u \nabla v \cdot \nabla u_0 \mathop{dx},
   \\
   L(v; \sigma_0, f)    &= \int_{\Omega} v \, f \mathop{dx}
   - \int_{\Omega} \nabla v \cdot \sigma_0 \mathop{dx}.

Then, two additional forms are created to compute the tangent C and
the gradient of :math:`u_0`. This situation shows up in plasticity and
other problems where certain quantities need to be computed elsewhere
(in user-defined functions).  The three forms using the standard
``FiniteElement`` (linear elements) can then be implemented as

::

  # NonlinearPoisson.ufl
  element = FiniteElement("Lagrange", triangle, 1)
  DG = FiniteElement("Discontinuous Lagrange", triangle, 0)
  sig = VectorElement("Discontinuous Lagrange", triangle, 0)

  v    = TestFunction(element)
  u    = TrialFunction(element)
  u0   = Coefficient(element)
  C    = Coefficient(DG)
  sig0 = Coefficient(sig)
  f    = Coefficient(element)

  a = v.dx(i)*C*u.dx(i)*dx + v.dx(i)*2*u0*u*u0.dx(i)*dx
  L = v*f*dx - dot(grad(v), sig0)*dx

and

::

  # Tangent.ufl
  element = FiniteElement("Lagrange", triangle, 1)
  DG = FiniteElement("Discontinuous Lagrange", triangle, 0)

  v = TestFunction(DG)
  u = TrialFunction(DG)
  u0= Coefficient(element)

  a = v*u*dx
  L = v*(1.0 + u0**2)*dx

and

::

  # Gradient.ufl
  element = FiniteElement("Lagrange", triangle, 1)
  DG = VectorElement("Discontinuous Lagrange", triangle, 0)

  v = TestFunction(DG)
  u = TrialFunction(DG)
  u0 = Coefficient(element)

  a = dot(v, u)*dx
  L = dot(v, (1.0 + u0**2)*grad(u0))*dx

The three forms can be implemented using the ``QuadratureElement``
in a similar fashion in which only the element declaration is different::

  # QE1NonlinearPoisson.ufl
  element = FiniteElement("Lagrange", triangle, 1)
  QE = FiniteElement("Quadrature", triangle, 2)
  sig = VectorElement("Quadrature", triangle, 2)

and

::

  # QE1Tangent.ufl
  element = FiniteElement("Lagrange", triangle, 1)
  QE = FiniteElement("Quadrature", triangle, 2)

and

::

  # QE1Gradient.ufl
  element = FiniteElement("Lagrange", triangle, 1)
  QE = VectorElement("Quadrature", triangle, 2)

Note that we use two points when declaring the ``QuadratureElement``. This
is because the RHS of ``Tangent.form`` is second order and therefore
we need two points for exact integration. Due to consistency issues,
when passing functions around between the forms, we also need to use
two points when declaring the ``QuadratureElement`` in the other forms.

Typical values of the relative residual for each Newton iteration for all
three approaches are shown in the table below. It is to be noted
that the convergence rate is quadratic as it should be for all three methods.

Relative residuals for each approach for linear elements::

  Iteration REF1      FE1      QE1
  ========= ====      ===      ===
  1         6.3e-02   6.3e-02  6.3e-02
  2         5.3e-04   5.3e-04  5.3e-04
  3         3.7e-08   3.7e-08  3.7e-08
  4         2.9e-16   2.9e-16  2.5e-16

However, if quadratic elements are used to interpolate the unknown field :math:`u`,
the order of all elements in the above forms is increased by 1. This influences
the convergence rate as seen in the table below. Clearly, using
the standard ``FiniteElement`` leads to a poor convergence whereas
the ``QuadratureElement`` still leads to quadratic convergence.

Relative residuals for each approach for quadratic elements::

  Iteration REF2      FE2      QE2
  ========= ====      ===      ===
  1         2.6e-01   3.9e-01  2.6e-01
  2         1.1e-02   4.6e-02  1.1e-02
  3         1.2e-05   1.1e-02  1.6e-05
  4         1.1e-11   7.2e-04  9.1e-09


More examples
=============

Feel free to send additional demo form files for your favourite PDE to
the UFL mailing list.

..
    %TODO: Modify rest of FFC example forms to UFL syntax and add here.
