"""Algorithm for replacing form arguments with 'stripped' versions.

In the stripped version, any data-carrying objects have been extracted to a mapping.
"""

from ufl.classes import Form, Integral
from ufl.classes import Argument, Coefficient, Constant
from ufl.classes import FunctionSpace, TensorProductFunctionSpace, MixedFunctionSpace
from ufl.classes import Mesh, MeshView, TensorProductMesh
from ufl.algorithms.replace import replace
from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction


class TerminalStripper(MultiFunction):
    """Terminal stripper."""

    def __init__(self):
        """Initialise."""
        super().__init__()
        self.mapping = {}

    def argument(self, o):
        """Apply to argument."""
        o_new = Argument(strip_function_space(o.ufl_function_space()),
                         o.number(), o.part())
        return self.mapping.setdefault(o, o_new)

    def coefficient(self, o):
        """Apply to coefficient."""
        o_new = Coefficient(strip_function_space(o.ufl_function_space()),
                            o.count())
        return self.mapping.setdefault(o, o_new)

    def constant(self, o):
        """Apply to constant."""
        o_new = Constant(strip_domain(o.ufl_domain()), o.ufl_shape,
                         o.count())
        return self.mapping.setdefault(o, o_new)

    expr = MultiFunction.reuse_if_untouched


def strip_terminal_data(o):
    """Return a new form where all terminals have been replaced by UFL-only equivalents.

    This function is useful for forms containing augmented UFL objects that
    hold references to large data structures. These objects are be extracted
    into the mapping allowing the form to be cached without leaking memory.

    Args:
        o: The object to be stripped. This must either be a Form or Integral

    Returns:
        A 2-tuple containing an equivalent UFL-only object and a mapping
        allowing the original form to be reconstructed using replace_terminal_data

    """
    # We need to keep track of two maps because integrals store references to the
    # domain and ``replace`` expects only a mapping containing ``Expr`` objects.
    if isinstance(o, Form):
        integrals = []
        expr_map = {}
        domain_map = {}
        for integral in o.integrals():
            itg, (emap, dmap) = strip_terminal_data(integral)
            integrals.append(itg)
            expr_map.update(emap)
            domain_map.update(dmap)
        return Form(integrals), (expr_map, domain_map)
    elif isinstance(o, Integral):
        handler = TerminalStripper()
        integrand = map_expr_dag(handler, o.integrand())
        domain = strip_domain(o.ufl_domain())
        # invert the mapping so it can be passed straight into replace_terminal_data
        expr_map = {v: k for k, v in handler.mapping.items()}
        domain_map = {domain: o.ufl_domain()}
        return o.reconstruct(integrand, domain=domain), (expr_map, domain_map)
    else:
        raise ValueError("Only Form or Integral inputs expected")


def replace_terminal_data(o, mapping):
    """Return a new form where the terminals have been replaced using the provided mapping.

    Args:
        o: The object to have its terminals replaced. This must either be a Form or Integral.
        mapping: A mapping suitable for reconstructing the form such as the one
            returned by strip_terminal_data.

    Returns:
        The new form.
    """
    if isinstance(o, Form):
        return Form([replace_terminal_data(itg, mapping) for itg in o.integrals()])
    elif isinstance(o, Integral):
        expr_map, domain_map = mapping
        integrand = replace(o.integrand(), expr_map)
        return o.reconstruct(integrand, domain=domain_map[o.ufl_domain()])
    else:
        raise ValueError("Only Form or Integral inputs expected")


def strip_function_space(function_space):
    """Return a new function space with all non-UFL information removed."""
    if isinstance(function_space, FunctionSpace):
        return FunctionSpace(strip_domain(function_space.ufl_domain()),
                             function_space.ufl_element())
    elif isinstance(function_space, TensorProductFunctionSpace):
        subspaces = [strip_function_space(sub) for sub in function_space.ufl_sub_spaces()]
        return TensorProductFunctionSpace(*subspaces)
    elif isinstance(function_space, MixedFunctionSpace):
        subspaces = [strip_function_space(sub) for sub in function_space.ufl_sub_spaces()]
        return MixedFunctionSpace(*subspaces)
    else:
        raise NotImplementedError(f"{type(function_space)} cannot be stripped")


def strip_domain(domain):
    """Return a new domain with all non-UFL information removed."""
    if isinstance(domain, Mesh):
        return Mesh(domain.ufl_coordinate_element(), domain.ufl_id())
    elif isinstance(domain, MeshView):
        return MeshView(strip_domain(domain.ufl_mesh()),
                        domain.topological_dimension(), domain.ufl_id())
    elif isinstance(domain, TensorProductMesh):
        meshes = [strip_domain(mesh) for mesh in domain.ufl_meshes()]
        return TensorProductMesh(meshes, domain.ufl_id())
    else:
        raise NotImplementedError(f"{type(domain)} cannot be stripped")
