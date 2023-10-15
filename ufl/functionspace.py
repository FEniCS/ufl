"""Types for representing function spaces."""

# Copyright (C) 2015-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Massimiliano Leoni, 2016
# Modified by Cecile Daversin-Catty, 2018

from ufl.core.ufl_type import UFLObject
from ufl.domain import join_domains
from ufl.duals import is_dual, is_primal

# Export list for ufl.classes
__all_classes__ = [
    "AbstractFunctionSpace",
    "FunctionSpace",
    "DualSpace",
    "MixedFunctionSpace",
    "TensorProductFunctionSpace",
]


class AbstractFunctionSpace(object):
    """Abstract function space."""

    def ufl_sub_spaces(self):
        """Return ufl sub spaces."""
        raise NotImplementedError(
            "Missing implementation of IFunctionSpace.ufl_sub_spaces in %s."
            % self.__class__.__name__
        )


class BaseFunctionSpace(AbstractFunctionSpace, UFLObject):
    """Base function space."""

    def __init__(self, domain, element):
        """Initialise."""
        if domain is None:
            # DOLFIN hack
            # TODO: Is anything expected from element.cell() in this case?
            pass
        else:
            try:
                domain_cell = domain.ufl_cell()
            except AttributeError:
                raise ValueError("Expected non-abstract domain for initalization of function space.")
            else:
                if element.cell() != domain_cell:
                    raise ValueError("Non-matching cell of finite element and domain.")

        AbstractFunctionSpace.__init__(self)
        self._ufl_domain = domain
        self._ufl_element = element

    def ufl_sub_spaces(self):
        """Return ufl sub spaces."""
        return ()

    def ufl_domain(self):
        """Return ufl domain."""
        return self._ufl_domain

    def ufl_element(self):
        """Return ufl element."""
        return self._ufl_element

    def ufl_domains(self):
        """Return ufl domains."""
        domain = self.ufl_domain()
        if domain is None:
            return ()
        else:
            return (domain,)

    def _ufl_hash_data_(self, name=None):
        """UFL hash data."""
        name = name or "BaseFunctionSpace"
        domain = self.ufl_domain()
        element = self.ufl_element()
        if domain is None:
            ddata = None
        else:
            ddata = domain._ufl_hash_data_()
        if element is None:
            edata = None
        else:
            edata = element._ufl_hash_data_()
        return (name, ddata, edata)

    def _ufl_signature_data_(self, renumbering, name=None):
        """UFL signature data."""
        name = name or "BaseFunctionSpace"
        domain = self.ufl_domain()
        element = self.ufl_element()
        if domain is None:
            ddata = None
        else:
            ddata = domain._ufl_signature_data_(renumbering)
        if element is None:
            edata = None
        else:
            edata = element._ufl_signature_data_()
        return (name, ddata, edata)

    def __repr__(self):
        """Representation."""
        r = "BaseFunctionSpace(%s, %s)" % (repr(self._ufl_domain),
                                           repr(self._ufl_element))
        return r


class FunctionSpace(BaseFunctionSpace, UFLObject):
    """Representation of a Function space."""

    _primal = True
    _dual = False

    def dual(self):
        """Get the dual of the space."""
        return DualSpace(self._ufl_domain, self._ufl_element)

    def _ufl_hash_data_(self):
        """UFL hash data."""
        return BaseFunctionSpace._ufl_hash_data_(self, "FunctionSpace")

    def _ufl_signature_data_(self, renumbering):
        """UFL signature data."""
        return BaseFunctionSpace._ufl_signature_data_(self, renumbering, "FunctionSpace")

    def __repr__(self):
        """Representation."""
        r = "FunctionSpace(%s, %s)" % (repr(self._ufl_domain),
                                       repr(self._ufl_element))
        return r


class DualSpace(BaseFunctionSpace, UFLObject):
    """Representation of a Dual space."""

    _primal = False
    _dual = True

    def __init__(self, domain, element):
        """Initialise."""
        BaseFunctionSpace.__init__(self, domain, element)

    def dual(self):
        """Get the dual of the space."""
        return FunctionSpace(self._ufl_domain, self._ufl_element)

    def _ufl_hash_data_(self):
        """UFL hash data."""
        return BaseFunctionSpace._ufl_hash_data_(self, "DualSpace")

    def _ufl_signature_data_(self, renumbering):
        """UFL signature data."""
        return BaseFunctionSpace._ufl_signature_data_(self, renumbering, "DualSpace")

    def __repr__(self):
        """Representation."""
        r = "DualSpace(%s, %s)" % (repr(self._ufl_domain),
                                   repr(self._ufl_element))
        return r


class TensorProductFunctionSpace(AbstractFunctionSpace, UFLObject):
    """Tensor product function space."""

    def __init__(self, *function_spaces):
        """Initialise."""
        AbstractFunctionSpace.__init__(self)
        self._ufl_function_spaces = function_spaces

    def ufl_sub_spaces(self):
        """Return ufl sub spaces."""
        return self._ufl_function_spaces

    def _ufl_hash_data_(self):
        """UFL hash data."""
        return ("TensorProductFunctionSpace",) \
            + tuple(V._ufl_hash_data_() for V in self.ufl_sub_spaces())

    def _ufl_signature_data_(self, renumbering):
        """UFL signature data."""
        return ("TensorProductFunctionSpace",) \
            + tuple(V._ufl_signature_data_(renumbering)
                    for V in self.ufl_sub_spaces())

    def __repr__(self):
        """Representation."""
        r = "TensorProductFunctionSpace(*%s)" % repr(self._ufl_function_spaces)
        return r


class MixedFunctionSpace(AbstractFunctionSpace, UFLObject):
    """Mixed function space."""

    def __init__(self, *args):
        """Initialise."""
        AbstractFunctionSpace.__init__(self)
        self._ufl_function_spaces = args
        self._ufl_elements = list()
        for fs in args:
            if isinstance(fs, BaseFunctionSpace):
                self._ufl_elements.append(fs.ufl_element())
            else:
                raise ValueError("Expecting BaseFunctionSpace objects")

        # A mixed FS is only primal/dual if all the subspaces are primal/dual"
        self._primal = all([is_primal(subspace)
                            for subspace in self._ufl_function_spaces])
        self._dual = all([is_dual(subspace)
                          for subspace in self._ufl_function_spaces])

    def ufl_sub_spaces(self):
        """Return ufl sub spaces."""
        return self._ufl_function_spaces

    def ufl_sub_space(self, i):
        """Return i-th ufl sub space."""
        return self._ufl_function_spaces[i]

    def dual(self, *args):
        """Return the dual to this function space.

        If no additional arguments are passed then a MixedFunctionSpace is
        returned whose components are the duals of the originals.

        If additional arguments are passed, these must be integers. In this
        case, the MixedFunctionSpace which is returned will have dual
        components in the positions corresponding to the arguments passed, and
        the original components in the other positions.
        """
        if args:
            spaces = [space.dual() if i in args else space
                      for i, space in enumerate(self._ufl_function_spaces)]
            return MixedFunctionSpace(*spaces)
        else:
            return MixedFunctionSpace(
                *[space.dual()for space in self._ufl_function_spaces]
            )

    def ufl_elements(self):
        """Return ufl elements."""
        return self._ufl_elements

    def ufl_element(self):
        """Return ufl element."""
        if len(self._ufl_elements) == 1:
            return self._ufl_elements[0]
        else:
            raise ValueError(
                "Found multiple elements. Cannot return only one. "
                "Consider building a FunctionSpace from a MixedElement "
                "in case of homogeneous dimension.")

    def ufl_domains(self):
        """Return ufl domains."""
        domainlist = []
        for s in self._ufl_function_spaces:
            domainlist.extend(s.ufl_domains())
        return join_domains(domainlist)

    def ufl_domain(self):
        """Return ufl domain."""
        domains = self.ufl_domains()
        if len(domains) == 1:
            return domains[0]
        elif domains:
            raise ValueError("Found multiple domains, cannot return just one.")
        else:
            return None

    def num_sub_spaces(self):
        """Return number of subspaces."""
        return len(self._ufl_function_spaces)

    def _ufl_hash_data_(self):
        """UFL hash data."""
        return ("MixedFunctionSpace",) \
            + tuple(V._ufl_hash_data_() for V in self.ufl_sub_spaces())

    def _ufl_signature_data_(self, renumbering):
        """UFL signature data."""
        return ("MixedFunctionSpace",) \
            + tuple(V._ufl_signature_data_(renumbering)
                    for V in self.ufl_sub_spaces())

    def __repr__(self):
        """Representation."""
        return f"MixedFunctionSpace(*{self._ufl_function_spaces})"
