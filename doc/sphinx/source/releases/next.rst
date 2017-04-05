===========================
Changes in the next release
===========================


Summary of changes
==================

.. note:: Developers should use this page to track and list changes
          during development. At the time of release, this page should
          be published (and renamed) to list the most important
          changes in the new release.

- Add the ``DirectionalSobolevSpace`` subclass of ``SobolevSpace``. This
  allows one to use spaces where elements have varying continuity in different
  spatial directions.
- Add ``sobolev_space`` methods for ``HDiv`` and ``HCurl`` finite elements.
- Add ``sobolev_space`` methods for ``TensorProductElement`` and ``EnrichedElement``.
  The smallest shared Sobolev space will be returned for enriched elements. For the
  tensor product elements, a ``DirectionalSobolevSpace`` is returned depending on the
  order of the spaces associated with the component elements.

Detailed changes
================

.. note:: At the time of release, make a verbatim copy of the
          ChangeLog here (and remove this note).
