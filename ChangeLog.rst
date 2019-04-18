Changelog
=========

2019.2.0.dev0
-------------

- No changes yet.

2019.1.0 (2019-04-17)
---------------------

- Remove scripts
- Remove LaTeX support (not functional)
- Add support for complex valued elements; complex mode
  is chosen by ``compute_form_data(form, complex_mode=True)`` typically
  by a form compiler; otherwise UFL language is agnostic to the choice
  of real/complex domain

2018.1.0 (2018-06-14)
---------------------

- Remove python2 support

2017.2.0 (2017-12-05)
---------------------

- Add geometric quantity ``CellDiameter`` defined as a set diameter
  of the cell, i.e., maximal distance between any two points of the
  cell; implemented on simplices and quads/hexes
- Rename internally used reference quantities
  ``(Cell|Facet)EdgeVectors`` to ``Reference(Cell|Facet)EdgeVectors``
- Add internally used quantites ``CellVertices``,
  ``(Cell|Facet)EdgeVectors`` which are physical-coordinates-valued;
  will be useful for further geometry lowering implementations
  for quads/hexes
- Implement geometry lowering of ``(Min|Max)(Cell|Facet)EdgeLength``
  for quads and hexes

2017.1.0.post1 (2017-09-12)
---------------------------

- Change PyPI package name to fenics-ufl.

2017.1.0 (2017-05-09)
---------------------

- Add the ``DirectionalSobolevSpace`` subclass of ``SobolevSpace``. This
  allows one to use spaces where elements have varying continuity in
  different spatial directions.
- Add ``sobolev_space`` methods for ``HDiv`` and ``HCurl`` finite
  elements.
- Add ``sobolev_space`` methods for ``TensorProductElement`` and
  ``EnrichedElement``.  The smallest shared Sobolev space will be
  returned for enriched elements. For the tensor product elements, a
  ``DirectionalSobolevSpace`` is returned depending on the order of the
  spaces associated with the component elements.

2016.2.0 (2016-11-30)
---------------------

- Add call operator syntax to ``Form`` to replace arguments and
  coefficients. This makes it easier to e.g. express the norm
  defined by a bilinear form as a functional. Example usage::

    # Equivalent to replace(a, {u: f, v: f})
    M = a(f, f)
    # Equivalent to replace(a, {f:1})
    c = a(coefficients={f:1})
- Add call operator syntax to ``Form`` to replace arguments and
  coefficients::

    a(f, g) == replace(a, {u: f, v: g})
    a(coefficients={f:1}) == replace(a, {f:1})
- Add ``@`` operator to ``Form``: ``form @ f == action(form, f)``
  (python 3.5+ only)
- Reduce noise in Mesh str such that ``print(form)`` gets more short and
  readable
- Fix repeated ``split(function)`` for arbitrary nested elements
- EnrichedElement: Remove ``+/*`` warning

  In the distant past, ``A + B => MixedElement([A, B])``.  The change
  that ``A + B => EnrichedElement([A, B])`` was made in ``d622c74`` (22
  March 2010).  A warning was introduced in ``fcbc5ff`` (26 March 2010)
  that the meaning of ``+`` had changed, and that users wanting a
  ``MixedElement`` should use ``*`` instead.  People have, presumably,
  been seeing this warning for 6 1/2 years by now, so it's probably safe
  to remove.
- Rework ``TensorProductElement`` implementation, replaces
  ``OuterProductElement``
- Rework ``TensorProductCell`` implementation, replaces
  ``OuterProductCell``
- Remove ``OuterProductVectorElement`` and ``OuterProductTensorElement``
- Add ``FacetElement`` and ``InteriorElement``
- Add ``Hellan-Herrmann-Johnson`` element
- Add support for double covariant and contravariant mappings in mixed
  elements
- Support discontinuous Taylor elements on all simplices
- Some more performance improvements
- Minor bugfixes
- Improve Python 3 support
- More permissive in integer types accepted some places
- Make ufl pass almost all flake8 tests
- Add bitbucket pipelines testing
- Improve documentation

2016.1.0 (2016-06-23)
---------------------

- Add operator A^(i,j) := as_tensor(A, (i,j))
- Updates to old manual for publishing on fenics-ufl.readthedocs.org
- Bugfix for ufl files with utf-8 encoding
- Bugfix in conditional derivatives to avoid inf/nan values in generated
  code. This bugfix may break ffc if uflacs is not used, to get around
  that the old workaround in ufl can be enabled by setting
  ufl.algorithms.apply_derivatives.CONDITIONAL_WORKAROUND = True
  at the top of your program.
- Allow sum([expressions]) where expressions are nonscalar by defining expr+0==expr
- Allow form=0; form -= other;
- Deprecate .cell(), .domain(), .element() in favour of .ufl_cell(),
	.ufl_domain(), .ufl_element(), in multiple classes, to allow
	closer integration with dolfin.
- Remove deprecated properties cell.{d,x,n,volume,circumradius,facet_area}.
- Remove ancient form2ufl script
- Add new class Mesh to replace Domain
- Add new class FunctionSpace(mesh, element)
- Make FiniteElement classes take Cell, not Domain.
- Large reworking of symbolic geometry pipeline
- Implement symbolic Piola mappings

1.6.0 (2015-07-28)
------------------

- Change approach to attaching __hash__ implementation to accomodate python 3
- Implement new non-recursive traversal based hash computation
- Allow derivative(M, ListTensor(<scalars>), ...) just like list/tuple works
- Add traits is_in_reference_frame, is_restriction, is_evaluation, is_differential
- Add missing linear operators to ArgumentDependencyExtractor
- Add _ufl_is_literal_ type trait
- Add _ufl_is_terminal_modifier_ type trait and Expr._ufl_terminal_modifiers_ list
- Add new types ReferenceDiv and ReferenceCurl
- Outer product element support in degree estimation
- Add TraceElement, InteriorElement, FacetElement, BrokenElement
- Add OuterProductCell to valid Real elements
- Add _cache member to form for use by external frameworks
- Add Sobolev space HEin
- Add measures dI,dO,dC for interface, overlap, cutcell
- Remove Measure constants
- Remove cell2D and cell3D
- Implement reference_value in apply_restrictions
- Rename point integral to vertex integral and kept ``*dP`` syntax
- Replace lambda functions in ufl_type with named functions for nicer
  stack traces
- Minor bugfixes, removal of unused code and cleanups

1.5.0 (2015-01-12)
------------------

- Require Python 2.7
- Python 3 support
- Change to py.test
- Rewrite parts of expression representation core, providing
  significant optimizations in speed and memory use, as well
  as a more elaborate type metadata system for internal use
- Use expr.ufl_shape instead of ufl.shape()
- Use expr.ufl_indices instead of ufl.indices(),
  returns tuple of free index ids, not Index objects
- Use expr.ufl_index_dimensions instead of ufl.index_dimensions(),
  returns tuple of dimensions ordered corresponding to expr.ufl_indices, not a dict
- Rewrite core algorithms for expression traversal
- Add new core algorithms map_expr_dag(), map_integrand_dag(),
  similar to python map() but applying a callable MultiFunction
  recursively to each Expr node, without Python recursion
- Highly recommend rewriting algorithms based on Transformer using
  map_expr_dag and MultiFunction, avoiding Python recursion overhead
- Rewrite core algorithms apply_derivatives, apply_restrictions
- Form signature is now computed without applying derivatives first,
  introducing smaller overhead on jit cache hits
- Use form.signature() to compute form signature
- Use form.arguments() instead of extract_arguments(form)
- Use form.coefficients() instead of extract_coefficients(form)
- Small improvement to str and latex output of expressions
- Allow diff(expr, coefficient) without wrapping coefficient in variable
- Add keywords to measures: dx(..., degree=3, rule="canonical")
- Introduce notation from the Periodic Table of the Finite Elements
- Introduce notation for FEEC families of elements: P-, P, Q-, S
- Experimental support for high-order geometric domains
- Algorithms for symbolic rewriting of geometric quantities (used by uflacs)
- Remove the *Constant* classes, using Coefficient with a Real element instead
- Add types for MinValue and MaxValue
- Disable automatic rewriting a+a->2*a, a*a->a**2, a/a->1, these are
  costly and the compiler should handle them instead
- Fix signature stability w.r.t. metadata dicts
- Minor bugfixes, removal of unused code and cleanups

1.4.0 (2014-06-02)
------------------

- New integral type custom_integral (``*dc``)
- Add analysis of which coefficients each integral actually uses to optimize assembly
- Improved svg rendering of cells and sobolevspaces in ipython notebook
- Add sobolev spaces, use notation "element in HCurl" (HCurl, HDiv, H1, H2, L2)
- Improved error checking of facet geometry in non-facet integrals
- Improved restriction handling, restricting continuous coefficients and constants is now optional
- Introduce notation from the Periodic Table of the Finite Elements (draft)
- Remove alias "Q" for quadrature element, use "Quadrature"
- New derivative type ReferenceGrad
- New discontinuous RT element
- New geometry types Jacobian, JacobianInverse, JacobianDeterminant
- New geometry types FacetJacobian, FacetJacobianInverse, FacetJacobianDeterminant
- New geometry types CellFacetJacobian, CellFacetJacobianInverse, CellFacetJacobianDeterminant
- New geometry types FacetOrigin, CellOrigin
- New geometry types CellCoordinate, FacetCoordinate
- New geometry types CellNormal, CellOrientation, QuadratureWeight
- Argument (and TestFunction, TrialFunction) now use absolute numbering f.number() instead of relative f.count()
- New syntax: integrand*dx(domain)
- New syntax: integrand*dx(1, domain=domain)
- New syntax: integrand*dx(1, subdomain_data=domain_data)
- Using domain instead of cell in many places.
- Deprecated notation 'cell.n', 'cell.x' etc.
- Recommended new notation: FacetNormal(domain)
- Experimental: Argument (and TestFunction, TrialFunction) now can have a specified part index for representing block systems
- Experimental: Domains can now be created with a Coefficient providing coordinates: Domain(Coefficient(VectorElement("CG", domain, 2)))
- Experimental: New concept Domain: domain = Domain(triangle, geometric_dimension=3, label="MyDomain")
- Various general optimizations
- Various minor bugfixes
- Various docstring improvements

1.3.0 (2014-01-07)
------------------

- Add cell_avg and facet_avg operators, can be applied to a Coefficient or Argument or restrictions thereof
- Fix bug in cofactor: now it is transposed the correct way.
- Add cell.min_facet_edge_length
- Add cell.max_facet_edge_length
- Simplify 0^f -> 0 if f is a non-negative scalar value
- Add atan2 function
- Allow form+0 -> form

1.2.0 (2013-03-24)
------------------

- NB! Using shapes such as (1,) and (1,1) instead of () for 1D tensor quantities I, x, grad(f)
- Add cell.facet_diameter
- Add new concept Domain
- Add new concept Region, which is the union of numbered subdomains
- Add integration over regions (which may be overlapping by sharing subdomains)
- Add integration over everywhere
- Add functions cosh, sinh, tanh, Max, Min
- Generalize jump(v,n) for rank(v) > 2
- Fix some minor bugs

1.1.0 (2013-01-07)
------------------

- Add support for pickling of expressions (thanks to Graham Markall)
- Add shorthand notation A**2 == inner(A, A), special cased for power 2.
- Add support for measure sum notation f*(dx(0) + dx(3)) == f*dx(0) + f*dx(3)
- Supporting code for bugfix in PyDOLFIN when comparing test/trial functions
- Remove support for tuple form notation as this was ambiguous
- Bugfix in quadrature degree estimation, never returning <0 now
- Remove use of cmp to accomodate removal from python 3

1.1-alpha-prerelease (2012-11-18)
---------------------------------

(Not released, snapshot archived with submission of UFL journal paper)
- Support adding 0 to forms, allowing sum([a])
- Major memory savings and optimizations.
- Some bugfixes.
- Add perp operator.
- Support nested tuple syntax like MixedElement((U,V),W)
- Allow outer(a, b, c, ...) by recursive application from left.
- Add simplification f/f -> 1
- Add operators <,>,<=,>= in place of lt,gt,le,ge

1.0.0 (2011-12-07)
------------------

- No changes since rc1.

1.0-rc1 (2011-11-22)
--------------------

- Added tests covering snippets from UFL chapter in FEniCS book
- Added more unit tests
- Added operators diag and diag_vector
- Added geometric quantities cell.surface_area and cell.facet_area
- Fixed rtruediv bug
- Fixed bug with derivatives of elements of type Real with unspecified cell

1.0-beta3 (2011-10-26)
----------------------

- Added nabla_grad and nabla_div operators
- Added error function erf(x)
- Added bessel functions of first and second kind, normal and modified,
  bessel_J(nu, x), bessel_Y(nu, x), bessel_I(nu, x), bessel_K(nu, x)
- Extended derivative() to allow indexed coefficient(s) as differentiation variable
- Made ``*Constant`` use the ``Real`` space instead of ``DG0``
- Bugfix in adjoint where test and trial functions were in different spaces
- Bugfix in replace where the argument to a grad was replaced with 0
- Bugfix in reconstruction of tensor elements
- Some other minor bugfixes

1.0-beta2 (2011-08-11)
----------------------

- Support c*form where c depends on a coefficient in a Real space

1.0-beta (2011-07-08)
---------------------

- Add script ufl-version
- Added syntax for associating an arbitrary domain data object with a measure:
	dss = ds[boundaries]; M = f*dss(1) + g*dss(2)
- Added new operators elem_mult, elem_div, elem_pow and elem_op for
  elementwise application of scalar operators to tensors of equal shape
- Added condition operators And(lhs,rhs) and Or(lhs,rhs) and Not(cond)
- Fixed support for symmetries in subelements of a mixed element
- Add support for specifying derivatives of coefficients to derivative()

0.9.1 (2011-05-16)
------------------

- Remove set_foo functions in finite element classes
- Change license from GPL v3 or later to LGPL v3 or later
- Change behavior of preprocess(), form.compute_form_data(), form_data.preprocessed_form
- Allowing grad, div, inner, dot, det, inverse on scalars
- Simplify Identity(1) -> IntValue(1) automatically
- Added Levi-Cevita symbol: e = PermutationSymbol(3); e[i,j,k]
- Fix bug with future division behaviour (ufl does not support floor division)
- Add subdomain member variables to form class
- Allow action on forms of arbitrary rank

0.9.0 (2011-02-23)
------------------

- Allow jump(Sigma, n) for matrix-valued expression Sigma
- Bug fix in scalar curl operator
- Bug fix in deviatoric operator

0.5.4 (2010-09-01)
------------------

- Bug fixes in PartExtracter
- Do not import x for coordinate
- Add Circumradius to Cell (Cell.circumradius)
- Add CellVolume to Cell (Cell.volume)

0.5.3 (2010-07-01)
------------------

- Rename ElementRestriction --> RestrictedElement
- Experimental import of x from tetrahedron
- Make lhs/rhs work for resrictions
- Redefine operator + for FiniteElements and replace + by *
- Rename ElementUnion -> EnrichedElement
- Add support for tan() and inverse trigonometric functions

0.5.2 (2010-02-15)
------------------

- Attach form data to preprocessed form, accessible by form.form_data()

0.5.1 (2010-02-03)
------------------

- Fix bug in propagate_restriction

0.5.0 (2010-02-01)
------------------

- Several interface changes in FormData class
- Introduce call preprocess(form) to be called at beginning of compilation
- Rename BasisFunction --> Argument
- Rename Function --> Coefficient

0.4.1 (2009-12-04)
------------------

- Redefine grad().T --> grad()
- New meaning of estimate_max_polynomial_degree
- New function estimate_total_polynomial_degree
- Allow degree = None and cell = None for elements

0.4.0 (2009-09-23)
------------------

- Extensions for ElementRestriction (restrict FiniteElement to Cell)
- Bug fix for lhs/rhs with list tensor types
- Add new log function set_prefix
- Add new log function log(level, message)
- Added macro cell integral ``*dE``
- Added mechanism to add additional integral types
- Added LiftingOperator and LiftingFunction
- Added ElementRestriction

0.3.0 (2009-05-28)
------------------

- Some critical bugfixes, in particular in differentiation.
- Added form operators "system" and "sensitivity_rhs".
- diff can take form as argument, applies to all integrands.
- Rudimentary precedence handling for better
  use of parentheses in str(expression).
- Added script ufl2py, mainly for debugging purposes.
- Crude implementation of estimate_max_polynomial_degree
  for quadrature degree estimation.
- Improved manual.

0.2.0 (2009-04-07)
------------------

- Initial release of UFL.

0.1.0 (unreleased)
------------------

- Unreleased development versions of UFL.
