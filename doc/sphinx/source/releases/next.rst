===========================
Changes in the next release
===========================


Summary of changes
==================

.. note:: Developers should use this page to track and list changes
          during development. At the time of release, this page should
          be published (and renamed) to list the most important
          changes in the new release.

- Deprecate ``.cell()``, ``.domain()``, ``.element()`` in favour of
  ``.ufl_cell()``, ``.ufl_domain()``, ``.ufl_element()``, in multiple
  classes, to allow closer integration with DOLFIN
- Remove deprecated properties
  ``cell.{d,x,n,volume,circumradius,facet_area}``
- Remove ancient ``form2ufl`` script
- Large reworking of symbolic geometry pipeline
- Implement symbolic Piola mappings
- ``OuterProductCell`` and ``OuterProductElement`` are merged into
  ``TensorProductCell`` and ``TensorProductElement`` respectively
- Better degree estimation for quadrilaterals
- Expansion rules for Q, DQ, RTCE, RTCF, NCE and NCF on tensor product
  cells
- Add discontinuous Taylor elements
- Add support for the mapping ``double covariant Piola`` in ``uflacs``
- Add support for the mapping ``double contravariant Piola`` in ``uflacs``
- Support for tensor-valued subelements in ``uflacs`` fixed
- Replacing ``Discontinuous Lagrange Trace`` with ``HDiv Trace`` and removing ``TraceElement``
- Assigning ``Discontinuous Lagrange Trace`` and ``DGT`` as aliases for ``HDiv Trace``

Detailed changes
================

.. note:: At the time of release, make a verbatim copy of the
          ChangeLog here (and remove this note).
