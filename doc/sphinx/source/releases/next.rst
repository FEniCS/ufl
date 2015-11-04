Changes in the next release of UFL
==================================

- Deprecate `.cell()`, `.domain()`, `.element()` in favour of
        `.ufl_cell()`, `.ufl_domain()`, `.ufl_element()`, in multiple
        classes, to allow closer integration with DOLFIN.

- Remove deprecated properties `cell.{d,x,n,volume,circumradius,facet_area}`.
- Remove ancient `form2ufl` script
- Large reworking of symbolic geometry pipeline
- Implement symbolic Piola mappings
