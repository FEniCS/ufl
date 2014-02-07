"Types for representing a cell, domain, and quantities computed from cell geometry."

# Copyright (C) 2008-2013 Martin Sandve Alnes
#
# This file is part of UFL.
#
# UFL is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# UFL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UFL. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg, 2009.
# Modified by Kristian B. Oelgaard, 2009
# Modified by Marie E. Rognes 2012
#
# First added:  2008-03-14
# Last changed: 2013-12-13

from ufl.log import warning, error, deprecate
from ufl.assertions import ufl_assert
from ufl.common import istr, EmptyDict
from ufl.terminal import Terminal
from ufl.protocols import id_or_none
from collections import defaultdict

# --- Expression node types

# Mapping from cell name to topological dimension
cellname2dim = {
    "cell0D": 0,
    "cell1D": 1,
    "cell2D": 2,
    "cell3D": 3,
    "vertex": 0,
    "interval": 1,
    "triangle": 2,
    "tetrahedron": 3,
    "quadrilateral": 2,
    "hexahedron": 3,
    }

# Mapping from cell name to facet name
cellname2facetname = {
    "cell0D": None,
    "cell1D": "vertex",
    "cell2D": "cell1D",
    "cell3D": "cell2D",
    "vertex": None,
    "interval": "vertex",
    "triangle": "interval",
    "tetrahedron": "triangle",
    "quadrilateral": "interval",
    "hexahedron": "quadrilateral"
    }

# Valid UFL cellnames
ufl_cellnames = tuple(sorted(cellname2dim.keys()))

class GeometricQuantity(Terminal):
    __slots__ = ("_domain",)
    def __init__(self, domain):
        Terminal.__init__(self)
        self._domain = as_domain(domain)

    def cell(self):
        return self._domain.cell()

    def domains(self):
        return (self._domain,)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # NB! Assuming all geometric quantities in here
        #     are are cellwise constant by default!
        return True

    def signature_data(self, domain_numbering):
        "Signature data of geometric quantities depend on the domain numbering."
        return (type(self).__name__,) + self._domain.signature_data(domain_numbering)

    def __hash__(self):
        return hash((type(self).__name__,) + self._domain.hash_data())

    def __eq__(self, other):
        return isinstance(other, self._uflclass) and other._domain == self._domain

class SpatialCoordinate(GeometricQuantity):
    "Representation of a spatial coordinate."
    __slots__ = ()
    def __init__(self, domain):
        GeometricQuantity.__init__(self, domain)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        return False

    def shape(self):
        return (self._domain.geometric_dimension(),)

    def evaluate(self, x, mapping, component, index_values):
        if component == ():
            if isinstance(x, (tuple,list)):
                return float(x[0])
            else:
                return float(x)
        else:
            return float(x[component[0]])

    def __str__(self):
        return "x"

    def __repr__(self):
        return "SpatialCoordinate(%r)" % self._domain

class LocalCoordinate(GeometricQuantity):
    "(EXPERIMENTAL) Representation of a local coordinate on the reference cell."
    __slots__ = ()
    def __init__(self, domain):
        GeometricQuantity.__init__(self, domain)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        return False

    def shape(self):
        return (self._domain.geometric_dimension(),)

    def evaluate(self, x, mapping, component, index_values):
        error("Symbolic evaluation of local coordinate not available.")

    def __str__(self):
        return "xi"

    def __repr__(self):
        return "LocalCoordinate(%r)" % self._domain

class CellBarycenter(GeometricQuantity):
    "Representation of the spatial barycenter coordinate of the cell."
    __slots__ = ()
    def __init__(self, domain):
        GeometricQuantity.__init__(self, domain)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        return True

    def shape(self):
        return (self._domain.geometric_dimension(),)

    def evaluate(self, x, mapping, component, index_values):
        error("Symbolic evaluation of the barycenter not available.")

    def __str__(self):
        return "cell_barycenter"

    def __repr__(self):
        return "CellBarycenter(%r)" % self._domain

class FacetBarycenter(GeometricQuantity):
    "Representation of the spatial barycenter coordinate of the facet."
    __slots__ = ()
    def __init__(self, domain):
        GeometricQuantity.__init__(self, domain)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        return True

    def shape(self):
        return (self._domain.geometric_dimension(),)

    def evaluate(self, x, mapping, component, index_values):
        error("Symbolic evaluation of the facet barycenter not available.")

    def __str__(self):
        return "facet_barycenter"

    def __repr__(self):
        return "FacetBarycenter(%r)" % self._domain

class Jacobian(GeometricQuantity):
    "Representation of the Jacobi of the mapping from local to global coordinates."
    __slots__ = ()
    def __init__(self, domain):
        GeometricQuantity.__init__(self, domain)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # FIXME: Not true for non-affine mappings
        return True

    def shape(self):
        return (self._domain.geometric_dimension(), self._domain.topological_dimension())

    def evaluate(self, x, mapping, component, index_values):
        error("Symbolic evaluation of geometry jacobi not available.")

    def __str__(self):
        return "J"

    def __repr__(self):
        return "Jacobian(%r)" % self._domain

class JacobianDeterminant(GeometricQuantity):
    "Representation of the determinant of the Jacobi of the mapping from local to global coordinates."
    __slots__ = ()
    def __init__(self, domain):
        GeometricQuantity.__init__(self, domain)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # FIXME: Not true for non-affine mappings
        return True

    def shape(self):
        return ()

    def evaluate(self, x, mapping, component, index_values):
        error("Symbolic evaluation of geometry jacobi determinant not available.")

    def __str__(self):
        return "detJ"

    def __repr__(self):
        return "JacobianDeterminant(%r)" % self._domain

class JacobianInverse(GeometricQuantity):
    "Representation of the (pseudo-)inverse of the Jacobi of the mapping from local to global coordinates."
    __slots__ = ()
    def __init__(self, domain):
        GeometricQuantity.__init__(self, domain)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # FIXME: Not true for non-affine mappings
        return True

    def shape(self):
        return (self._domain.topological_dimension(), self._domain.geometric_dimension())

    def evaluate(self, x, mapping, component, index_values):
        error("Symbolic evaluation of inverse geometry jacobi not available.")

    def __str__(self):
        return "K"

    def __repr__(self):
        return "JacobianInverse(%r)" % self._domain

class FacetNormal(GeometricQuantity):
    "Representation of a facet normal."
    __slots__ = ()
    def __init__(self, domain):
        GeometricQuantity.__init__(self, domain)

    def shape(self):
        return (self._domain.geometric_dimension(),)

    def __str__(self):
        return "n"

    def __repr__(self):
        return "FacetNormal(%r)" % self._domain

class CellNormal(GeometricQuantity):
    "Representation of a cell normal, for cells of tdim=gdim-1."
    __slots__ = ()
    def __init__(self, domain):
        GeometricQuantity.__init__(self, domain)

    def shape(self):
        return (self._domain.geometric_dimension(),)

    def __str__(self):
        return "cell_normal"

    def __repr__(self):
        return "CellNormal(%r)" % self._domain

class CellVolume(GeometricQuantity):
    "Representation of a cell volume."
    __slots__ = ()
    def __init__(self, domain):
        GeometricQuantity.__init__(self, domain)

    def shape(self):
        return ()

    def __str__(self):
        return "volume"

    def __repr__(self):
        return "CellVolume(%r)" % self._domain

class Circumradius(GeometricQuantity):
    "Representation of the circumradius of a cell."
    __slots__ = ()
    def __init__(self, domain):
        GeometricQuantity.__init__(self, domain)

    def shape(self):
        return ()

    def __str__(self):
        return "circumradius"

    def __repr__(self):
        return "Circumradius(%r)" % self._domain

class CellSurfaceArea(GeometricQuantity):
    "Representation of the total surface area of a cell."
    __slots__ = ()
    def __init__(self, domain):
        GeometricQuantity.__init__(self, domain)

    def shape(self):
        return ()

    def __str__(self):
        return "surfacearea"

    def __repr__(self):
        return "CellSurfaceArea(%r)" % self._domain

class FacetArea(GeometricQuantity):
    "Representation of the area of a cell facet."
    __slots__ = ()
    def __init__(self, domain):
        GeometricQuantity.__init__(self, domain)

    def shape(self):
        return ()

    def __str__(self):
        return "facetarea"

    def __repr__(self):
        return "FacetArea(%r)" % self._domain

class FacetDiameter(GeometricQuantity):
    """(EXPERIMENTAL) Representation of the diameter of a facet.

    This is not yet defined.
    """
    __slots__ = ()
    def __init__(self, domain):
        GeometricQuantity.__init__(self, domain)

    def shape(self):
        return ()

    def __str__(self):
        return "facetdiameter"

    def __repr__(self):
        return "FacetDiameter(%r)" % self._domain

class MinFacetEdgeLength(GeometricQuantity):
    "Representation of the minimum edge length of a facet."
    __slots__ = ()
    def __init__(self, domain):
        GeometricQuantity.__init__(self, domain)

    def shape(self):
        return ()

    def __str__(self):
        return "minfacetedgelength"

    def __repr__(self):
        return "MinFacetEdgeLength(%r)" % self._domain

class MaxFacetEdgeLength(GeometricQuantity):
    "Representation of the maximum edge length of a facet."
    __slots__ = ()
    def __init__(self, domain):
        GeometricQuantity.__init__(self, domain)

    def shape(self):
        return ()

    def __str__(self):
        return "maxfacetedgelength"

    def __repr__(self):
        return "MaxFacetEdgeLength(%r)" % self._domain


# --- Basic cell representation classes

# TODO: Remove this deprecated part after a release or two.
class DeprecatedGeometryProperties(object):
    __slots__ = ()

    @property
    def x(self):
        "UFL geometry value: The global spatial coordinates."
        deprecate("cell.x is deprecated, please use SpatialCoordinate(domain) instead")
        return SpatialCoordinate(as_domain(self))

    @property
    def xi(self):
        "UFL geometry value: The local spatial coordinates."
        deprecate("cell.xi is deprecated, please use LocalCoordinate(domain) instead")
        return LocalCoordinate(as_domain(self))

    @property
    def J(self):
        "UFL geometry value: The Jacobi of the local to global coordinate mapping."
        deprecate("cell.J is deprecated, please use Jacobian(domain) instead")
        return Jacobian(as_domain(self))

    @property
    def detJ(self):
        "UFL geometry value: The determinant of the Jacobi of the local to global coordinate mapping."
        deprecate("cell.detJ is deprecated, please use JacobianDeterminant(domain) instead")
        return JacobianDeterminant(as_domain(self))

    @property
    def Jinv(self):
        "UFL geometry value: The inverse of the Jacobi of the local to global coordinate mapping."
        deprecate("cell.Jinv is deprecated, please use JacobianInverse(domain) instead")
        return JacobianInverse(as_domain(self))

    @property
    def n(self):
        "UFL geometry value: The facet normal on the cell boundary."
        deprecate("cell.n is deprecated, please use FacetNormal(domain) instead")
        return FacetNormal(as_domain(self))

    @property
    def volume(self):
        "UFL geometry value: The volume of the cell."
        deprecate("cell.volume is deprecated, please use CellSize(domain) instead")
        return CellVolume(as_domain(self))

    @property
    def circumradius(self):
        "UFL geometry value: The circumradius of the cell."
        return Circumradius(as_domain(self))

    @property
    def facet_diameter(self):
        "UFL geometry value: The diameter of a facet of the cell."
        deprecate("cell.facet_diameter is deprecated, please use FacetDiameter(domain) instead")
        return FacetDiameter(as_domain(self))

    @property
    def min_facet_edge_length(self):
        "UFL geometry value: The minimum edge length of a facet of the cell."
        deprecate("cell.min_facet_edge_length is deprecated, please use MinFacetEdgeLength(domain) instead")
        return MinFacetEdgeLength(as_domain(self))

    @property
    def max_facet_edge_length(self):
        "UFL geometry value: The maximum edge length of a facet of the cell."
        deprecate("cell.max_facet_edge_length is deprecated, please use MaxFacetEdgeLength(domain) instead")
        return MaxFacetEdgeLength(as_domain(self))

    @property
    def facet_area(self):
        "UFL geometry value: The area of a facet of the cell."
        deprecate("cell.facet_area is deprecated, please use FacetArea(domain) instead")
        return FacetArea(as_domain(self))

    @property
    def surface_area(self):
        "UFL geometry value: The total surface area of the cell."
        deprecate("cell.surface_area is deprecated, please use SurfaceArea(domain) instead")
        return CellSurfaceArea(as_domain(self))

    @property
    def d(self):
        """The dimension of the cell.

        Only valid if the geometric and topological dimensions are the same.
        """
        deprecate("cell.d is deprecated, please use one of cell.topological_dimension(), cell.geometric_dimension(), domain.topological_dimension() or domain.geometric_dimension() instead.")
        ufl_assert(self.topological_dimension() == self.geometric_dimension(),
                   "Cell.d is undefined when geometric and"+\
                   "topological dimensions are not the same.")
        return self.geometric_dimension()


# TODO: When we have removed DeprecatedGeometryProperties, extract Cell, Domain to another file.
class Cell(DeprecatedGeometryProperties):
    "Representation of a finite element cell."
    __slots__ = ("_cellname",
                 "_geometric_dimension",
                 "_topological_dimension"
                 )
    def __init__(self, cellname, geometric_dimension=None, topological_dimension=None):
        "Initialize basic cell description."

        # The cellname must be among the known ones,
        # so we can find the known dimension
        dim = cellname2dim[cellname]

        # The topological dimension is defined by the cell type,
        # or given for product cells
        if topological_dimension is None:
            tdim = dim
        else:
            tdim = topological_dimension

        # The geometric dimension defaults to equal the topological
        # dimension if undefined
        if geometric_dimension is None:
            gdim = tdim
        else:
            gdim = geometric_dimension

        # Validate dimensions
        ufl_assert(isinstance(gdim, int),
                   "Expecting integer dimension, not '%r'" % (gdim,))
        ufl_assert(isinstance(tdim, int),
                   "Expecting integer dimension, not '%r'" % (tdim,))
        ufl_assert(tdim <= gdim,
                   "Topological dimension cannot be larger than geometric dimension.")

        # ... Finally store validated data
        self._cellname = cellname
        self._topological_dimension = tdim
        self._geometric_dimension = gdim

    def geometric_dimension(self):
        "Return the dimension of the space this cell is embedded in."
        return self._geometric_dimension

    def topological_dimension(self):
        "Return the dimension of the topology of this cell."
        return self._topological_dimension

    def cellname(self):
        "Return the cellname of the cell."
        return self._cellname

    def facet_cellname(self):
        "Return the cellname of the facet of this cell."
        facet_cellname = cellname2facetname.get(self._cellname)
        ufl_assert(facet_cellname is not None,
                   "Name of facet cell not available for cell type %s." % self._cellname)
        return facet_cellname

    def __eq__(self, other):
        if not isinstance(other, Cell):
            return False
        s = (self._geometric_dimension, self._topological_dimension, self._cellname)
        o = (other._geometric_dimension, other._topological_dimension, other._cellname)
        return s == o

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if not isinstance(other, Cell):
            return False
        s = (self._geometric_dimension, self._topological_dimension, self._cellname)
        o = (other._geometric_dimension, other._topological_dimension, other._cellname)
        return s < o

    def __hash__(self):
        return hash(repr(self))

    def __str__(self):
        return "<%s cell in %sD>" % (istr(self._cellname),
                                     istr(self._geometric_dimension))

    def __repr__(self):
        return "Cell(%r, %r)" % (self._cellname, self._geometric_dimension)

    def _repr_svg_(self):
        n = self.cellname()
        svg = '<svg xmlns="http://www.w3.org/2000/svg" version="1.1">\n<polyline points="%s" style="fill:none;stroke:black;stroke-width:3" />\n</svg>'
        if n == "interval":
            svg = svg % '0,0, 200,0'
        elif n == "triangle":
            svg = svg % '0,200 200,200 0,0 0,200'
        elif n == "quadrilateral":
            svg = svg % '0,200 200,200 200,0 0,0 0,200'
        else:
            svg = None
        return svg

class ProductCell(Cell):
    __slots__ = ("_cells",)
    def __init__(self, *cells):
        cells = tuple(as_cell(cell) for cell in cells)
        gdim = sum(cell.geometric_dimension() for cell in cells)
        tdim = sum(cell.topological_dimension() for cell in cells)
        Cell.__init__("product", gdim, tdim)
        self._cells = tuple(cells)

    def sub_cells(self):
        "Return list of cell factors."
        return self._cells

    def facet_cellname(self):
        "Return the cellname of the facet of this cell."
        error("Makes no sense for product cell.")

    def __eq__(self, other):
        if not isinstance(other, ProductCell):
            return False
        return self._cells == other._cells

    def __lt__(self, other):
        if not isinstance(other, ProductCell):
            return False
        return self._cells < other._cells

    def __repr__(self):
        return "ProductCell(*%r)" % (self._cells,)

class Domain(object):
    __slots__ = ("_cell",
                 "_geometric_dimension",
                 "_topological_dimension",
                 "_label",
                 "_data",
                 )
    def __init__(self, cell=None,
                 geometric_dimension=None,
                 topological_dimension=None,
                 label=None, data=None):

        # FIXME: I'm not sure what I tried to do with the dimensions here...
        #        A cell must have same dimensions as the domain, right?
        #        Or does it make sense to have e.g. interval cells in 3D from a triangle mesh in 3D?
        #        How does product cells work out?

        # If no cell name is given, create generic nD cell
        if cell is None:
            cell = "cell%dD" % topological_dimension
        # If cell name is given, create Cell
        if isinstance(cell, str):
            cell = Cell(cell, geometric_dimension=geometric_dimension)
        # Now we should have a Cell or something went wrong
        ufl_assert(isinstance(cell, Cell), "Failed to construct a Cell from input arguments.")
        self._cell = cell

        self._geometric_dimension = (self._cell.geometric_dimension()
                                     if geometric_dimension is None
                                     else geometric_dimension)
        self._topological_dimension = (self._cell.topological_dimension()
                                       if topological_dimension is None
                                       else topological_dimension)

        # Sanity checks
        ufl_assert(isinstance(self._geometric_dimension, int),
                   "Expecting integer geometric dimension.")
        ufl_assert(isinstance(self._topological_dimension, int),
                   "Expecting integer topological dimension.")
        ufl_assert(self._topological_dimension <= self._geometric_dimension,
                   "Topological dimension cannot be greater than geometric dimension.")
        ufl_assert(self._topological_dimension >= 0,
                   "Topological dimension must be non-negative.")
 
        self._label = label
        ufl_assert(self._label is None or isinstance(self._label, str),
                   "Expecting None or str for label.")

        self._data = data
        ufl_assert(self._data is None or hasattr(data, "ufl_id"),
                   "Expecting data object to implement ufl_id().")

    def reconstruct(self, label=None, data=None):
        "Create a new Domain object with possibly changed label or data."
        if label is None:
            label = self.label()
        if data is None:
            data = self.data()
        return Domain(cell=self.cell(),
                      geometric_dimension=self.geometric_dimension(),
                      topological_dimension=self.topological_dimension(),
                      label=label, data=data)

    def geometric_dimension(self):
        "Return the dimension of the space this domain is embedded in."
        return self._geometric_dimension

    def topological_dimension(self):
        "Return the dimension of the topology of this domain."
        return self._topological_dimension

    def cell(self):
        "Return the cell this domain is defined in terms of."
        return self._cell

    def label(self):
        "Return the label identifying this domain. None means no label has been set."
        return self._label

    def data(self):
        "Return attached data object."
        return self._data

    def domain_numbering_key(self):
        """Key used to look up in domain_numbering given to signature_data.

        TODO: This design is getting hard to follow,
        a consequence of a number of compatibility hacks.
        Consider simplifying some of the new features to make
        the combination of features and compatibility smoother.
        """
        key = (self._cell.cellname(),
               self._geometric_dimension,
               self._topological_dimension,
               self._label)
        return key

    def signature_data(self, domain_numbering):
        "Signature data of domain depend on the global domain numbering."
        key = self.domain_numbering_key()
        data = key[:-1] + (domain_numbering[key],)
        return data

    # TODO: Add this signature to terminal expr types?
    def hash_data(self):
        # Including only id of data here.
        # If this is a problem in pydolfin, the user will just have
        # to create explicit Domain objects to avoid problems.
        return (self._label,
                self._geometric_dimension,
                self._topological_dimension,
                self._cell,
                id_or_none(self._data))

    def __hash__(self):
        return hash(self.hash_data())

    def __eq__(self, other):
        return type(self) == type(other) and self.hash_data() == other.hash_data()

    def __lt__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return self.hash_data() < other.hash_data()

    def __str__(self):
        s = (self._label, self._topological_dimension,
             self._geometric_dimension, self._cell)
        return "<Domain '%s', a %dD domain embedded in %dD, built from %s cells>" % s

    def reconstruction_signature(self):
        """Format as string for evaluation as Python object.

        For use with cross language frameworks, stored in generated code
        and evaluated later in Python to reconstruct this object.

        This differs from repr in that it does not include domain
        label and data, which must be reconstructed or supplied by other means.
        """
        s = (self._cell, self._geometric_dimension, self._topological_dimension)
        return "Domain(cell=%r, geometric_dimension=%r, topological_dimension=%r)" % s

    def __repr__(self):
        if self._data is None:
            d = None
        else:
            d = "<data with id %s>" % id_or_none(self._data)
        s = (self._cell, self._geometric_dimension,
             self._topological_dimension, self._label, d)
        return "Domain(cell=%r, geometric_dimension=%r, topological_dimension=%r, label=%r, data=%r)" % s


class OverlapDomain(Domain):
    """WARNING: This is work in progress, design is in no way completed."""
    __slots__ = ("_child_domains",)
    def __init__(self, domain1, domain2, label=None, data=None):
        # Check domain compatibility
        ufl_assert(domain1.cell() == domain2.cell(),
                   "Cell mismatch in overlap domain.")
        ufl_assert(domain1.geometric_dimension() == domain2.geometric_dimension(),
                   "Dimension mismatch in overlap domain.")
        ufl_assert(domain1.topological_dimension() == domain2.topological_dimension(),
                   "Dimension mismatch in overlap domain.")

        # Get the right properties of this domain
        cell = domain1.cell()
        gdim = domain1.geometric_dimension()
        tdim = domain1.topological_dimension()

        # Initialize parent class
        Domain.__init__(self, cell, gdim, tdim, label=label, data=data)

        # Save child domains for later
        self._child_domains = (domain1, domain2)

    def child_domains(self):
        return self._child_domains

class IntersectionDomain(Domain):
    """WARNING: This is work in progress, design is in no way completed."""
    __slots__ = ("_child_domains",)
    def __init__(self, domain1, domain2, label=None, data=None):
        # Check domain compatibility
        ufl_assert(domain1.cell() == domain2.cell(),
                   "Cell mismatch in overlap domain.")
        ufl_assert(domain1.geometric_dimension() == domain2.geometric_dimension(),
                   "Dimension mismatch in overlap domain.")
        ufl_assert(domain1.topological_dimension() == domain2.topological_dimension(),
                   "Dimension mismatch in overlap domain.")

        # Get the right properties of this domain
        gdim = domain1.geometric_dimension()
        tdim = domain1.topological_dimension()-1
        cell = Cell(domain1.cell().facet_cellname(), gdim)
        ufl_assert(cell.topological_dimension() == tdim)

        # Initialize parent class
        Domain.__init__(self, cell, gdim, tdim, label=label, data=data)

        # Save child domains for later
        self._child_domains = (domain1, domain2)

    def child_domains(self):
        return self._child_domains

# --- Utility conversion functions

def as_cell(cell):
    """Convert any valid object to a Cell (in particular, cellname string),
    or return cell if it is already a Cell."""
    if isinstance(cell, Cell):
        return cell
    elif hasattr(cell, "ufl_cell"):
        return cell.ufl_cell()
    elif isinstance(cell, str):
        return Cell(cell)
    else:
        error("Invalid cell %s." % cell)

def as_domain(domain):
    """Convert any valid object to a Domain (in particular, cell or cellname string),
    or return domain if it is already a Domain."""
    if isinstance(domain, Domain):
        return domain
    elif hasattr(domain, "ufl_domain"):
        return domain.ufl_domain()
    else:
        return Domain(as_cell(domain))

def join_domain_data(domain_datas):
    newdata = {}
    for data in domain_datas:
        for k,v in data.iteritems():
            nv = newdata.get(k)
            if nv is None:
                # New item, just add it
                newdata[k] = v
            elif v is not None:
                id1 = id_or_none(nv)
                id2 = id_or_none(v)
                if id1 != id2:
                    error("Found multiple data objects with key %s." % k)
    return newdata

def check_domain_compatibility(domains):
    # Validate that the domains are the same except for possibly the data
    cell = domains[0].cell()
    gdim = domains[0].geometric_dimension()
    tdim = domains[0].topological_dimension()
    for dom in domains[1:]:
        if dom.cell() != cell:
            error("Cell mismatch between domains.")
        if dom.geometric_dimension() != gdim:
            error("Geometric dimension mismatch between domains.")
        if dom.topological_dimension() != tdim:
            error("Topological dimension mismatch between domains.")

def join_domains(domains):
    """Take a list of Domains and return a list with only unique domain objects.

    Checks that domains with the same label are compatible,
    and allows data to be None or 
    """
    # Ignore Nones in input domains
    domains = [domain for domain in domains if domain is not None]

    # Build lists of domain objects with same label
    label2domlist = defaultdict(list)
    for domain in domains:
        label2domlist[domain.label()].append(domain)

    # Extract None list from this dict, map to label but only if only one exists
    if None in label2domlist:
        none_domains = {}
        if len(label2domlist) == 1:
            pass
        elif len(label2domlist) == 2:
            none_domains = label2domlist[None]
            del label2domlist[None]
            key, = label2domlist.keys()
            label2domlist[key].extend(none_domains)
        else:
            error("Ambiguous mapping of domains with label None to multiple domains with different labels.")
    else:
        none_domains = {}

    # Join domain data to get a list with only one domain for each label
    newdomains = []
    for label in sorted(label2domlist.keys()):
        domlist = label2domlist[label]
        if len(domlist) == 1:
            dom, = domlist
        else:
            # Validate that the domains are the same except for possibly the data
            check_domain_compatibility(domains)

            # Pick first non-None data object
            for dom in domlist:
                newdata = dom.data()
                if newdata is not None:
                    break
            cell = dom.cell()
            gdim = dom.geometric_dimension()
            tdim = dom.topological_dimension()

            # Validate that data ids match if present
            if newdata is not None:
                data_ids = [id_or_none(dom.data()) for dom in domlist]
                data_ids = set(i for i in data_ids if i is not None)
                if len(data_ids) > 1:
                    error("Found data objects with different ids in domains with same label.")

            # Construct a new domain object with fully completed data
            dom = Domain(cell=cell,
                         geometric_dimension=gdim,
                         topological_dimension=tdim,
                         label=label,
                         data=newdata)
        newdomains.append(dom)
    return newdomains

def extract_domains(expr):
    # FIXME: Consider components?
    from ufl.algorithms.traversal import traverse_terminals
    domainlist = []
    for t in traverse_terminals(expr):
        domainlist.extend(t.domains())
    return sorted(join_domains(domainlist))
