IDEAS
=====

* A ufl.Domain object represents a domain that can be integrated over:

    class Domain(Countable):
        def __init__(self, gdim, tdim, cellname, data=None, count=None):
            ...

    data is intended to be a dict, holding references to a mesh and mesh functions for the domain markers.
    The Domain needs to be:

      - Comparable (all members equal)
      - Sortable (by count, then by other members if not the same object)
      - Hashable (using id(data) or taking id of values in data, so we can do mapping[domain] = foo)
      - Pretty-printable

* In dolfin, the count can be set from the mesh.id() in dolfin mesh.ufl_domain():
      class Mesh:
          def ufl_domain(self, cell_domains=None, ...):
              data = { "mesh": self, "cell_domains": cell_domains, ... }
              return ufl.Domain(cell=self.ufl_cell(), gdim=self.geometry().dim(), tdim=self.topology().dim(), data=data, count=-self.id())

  (The dolfin mesh.id() from dolfin Variable is increasing for each mesh construction,
  making it similar to the ufl Countable concept.)

* Different types have relations to one or more Domains (implement 'def domains(self)' in all these):

  - A helper function unique_domains(domains) can be very useful in the below classes!

  - A Measure object has a single integration_domain() which can be None.

  - FiniteElement objects are currently related to only one domain, so domains() will always return a one item collection.

  - A future CompositeFiniteElement object may be related to several domains() (although subelements relate to only one domain each?).

  - FormArgument objects are related to the domains() of their element.

  - GeometricQuantity classes are related to only one domain. Example:

      x1 = SpatialCoordinate(D1)
      x2 = SpatialCoordinate(D2)
      assert x1.domains() == (D1,)
      assert x2.domains() == (D2,)

  - An Expr is related to the set of domains occuring in its terminals. Example:

      dist = (x1-x2)**2 # Should compute to zero, although x1 and x2 should be computed from separate domain cells in generated code
      assert dist.domains() == (D1, D2)

  - An Integral has a single integration_domain(), but potentially multiple domains() from its integrand (domains() includes the integration_domain()).

  - A Form has integration_domains() bounded by number of Integrals, and potentially multiple domains() from all its integrands.

* In expr*measure, create an Integral with an associated Domain
  to integrate over, taken from either expr or measure:

  -  if measure has a domain, use that.

  -  if measure has no domain, take it from expr.domains()

     + if len(expr.domains()) == 1, just take the one and we're ok.

     + if len(expr.domains()) != 1, want to trigger undefined error: cannot determine integration domain

     + however we can't trigger undefined error because dolfin should keep compatibility with

           assemble(Constant(1)*dx, mesh=mesh)
           assemble(f*g*dx, mesh=mesh1) # f from mesh2, g from mesh3

       so we'll instead allow the Integral to have None as the integration domain
       and postpone error checking to assemble/jit.

     + if len(integral.domains()) > 1, we're entering multi-mesh territory. To be continued!

* Assembling a form with None as the domain triggers undefined
  error UNLESS a mesh is provided, in which case the form is
  reconstructed with a domain as a preprocessing step.

* Jitting a form with None as the domain triggers undefined error.

* Implement this in dolfin as:

  assemble(incomplete_form, data=foo) =>
  assemble(reconstruct_form(incomplete_form, data=foo))

  e.g.

  assemble(incomplete_form, mesh=foo) =>
  assemble(reconstruct_form(incomplete_form, mesh=foo))

  assemble(incomplete_form, facet_domains=foo) =>
  assemble(reconstruct_form(incomplete_form, facet_domains=foo))

* FIXME: Figure out which places are left where we need undefined cells, that's a major headache!

  - dolfin.Expression will always have undefined element/cells

  - dolfin.Constant has no cell, but that's ok and should be made a more common case with fundamental support in UFC for global dofs.

  - What will happen when someone does:

        C = Overlap(A,B)
        M = expression*C.dw()

    Should the expression then be interpolated to FunctionSpace(C.mesh(), "CG", 1)?

ACTION
======


