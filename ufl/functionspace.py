# -*- coding: utf-8 -*-
"Types for representing function spaces."

# Copyright (C) 2015-2016 Martin Sandve Alnæs
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
# Modified by Massimiliano Leoni, 2016

from ufl.utils.py23 import as_native_str
from ufl.utils.py23 import as_native_strings
from ufl.log import error
from ufl.core.ufl_type import attach_operators_from_hash_data
from ufl.domain import join_domains
from ufl.finiteelement import MixedElement

# Export list for ufl.classes
__all_classes__ = as_native_strings([
    "AbstractFunctionSpace",
    "FunctionSpace",
    "MixedFunctionSpace",
    "TensorProductFunctionSpace",
])


class AbstractFunctionSpace(object):
    def ufl_sub_spaces(self):
        raise NotImplementedError("Missing implementation of IFunctionSpace.ufl_sub_spaces in %s." % self.__class__.__name__)


@attach_operators_from_hash_data
class FunctionSpace(AbstractFunctionSpace):
    def __init__(self, domain, element):
        if domain is None:
            # DOLFIN hack
            # TODO: Is anything expected from element.cell() in this case?
            pass
        else:
            try:
                domain_cell = domain.ufl_cell()
            except AttributeError:
                error("Expected non-abstract domain for initalization of function space.")
            else:
                if element.cell() != domain_cell:
                    error("Non-matching cell of finite element and domain.")

        AbstractFunctionSpace.__init__(self)
        self._ufl_domain = domain
        self._ufl_element = element

    def ufl_sub_spaces(self):
        "Return ufl sub spaces."
        return ()

    def ufl_domain(self):
        "Return ufl domain."
        return self._ufl_domain

    def ufl_element(self):
        "Return ufl element."
        return self._ufl_element

    def ufl_domains(self):
        "Return ufl domains."
        domain = self.ufl_domain()
        if domain is None:
            return ()
        else:
            return (domain,)

    def _ufl_hash_data_(self):
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
        return ("FunctionSpace", ddata, edata)

    def _ufl_signature_data_(self, renumbering):
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
        return ("FunctionSpace", ddata, edata)

    def __repr__(self):
        r = "FunctionSpace(%s, %s)" % (repr(self._ufl_domain), repr(self._ufl_element))
        return as_native_str(r)


@attach_operators_from_hash_data
class MixedFunctionSpace(AbstractFunctionSpace):
    def __init__(self, *function_spaces):
        AbstractFunctionSpace.__init__(self)
        self._ufl_function_spaces = function_spaces
        self._ufl_element = MixedElement(*[fs.ufl_element() for fs in function_spaces])

    def ufl_sub_spaces(self):
        "Return ufl sub spaces."
        return self._ufl_function_spaces

    def ufl_element(self):
        "Return ufl element."
        return self._ufl_element

    def ufl_domains(self):
        "Return ufl domains."
        domainlist = []
        for s in self._ufl_function_spaces:
            domainlist.extend(s.ufl_domains())
        return join_domains(domainlist)

    def ufl_domain(self):
        "Return ufl domain."
        domains = self.ufl_domains()
        if len(domains) == 1:
            return domains[0]
        elif domains:
            error("Found multiple domains, cannot return just one.")
        else:
            return None

    def _ufl_hash_data_(self):
        return ("MixedFunctionSpace",) + tuple(V._ufl_hash_data_() for V in self.ufl_sub_spaces())

    def _ufl_signature_data_(self, renumbering):
        return ("MixedFunctionSpace",) + tuple(V._ufl_signature_data_(renumbering) for V in self.ufl_sub_spaces())

    def __repr__(self):
        r = "MixedFunctionSpace(*%s)" % repr(self._ufl_function_spaces)
        return as_native_str(r)


@attach_operators_from_hash_data
class TensorProductFunctionSpace(AbstractFunctionSpace):
    def __init__(self, *function_spaces):
        AbstractFunctionSpace.__init__(self)
        self._ufl_function_spaces = function_spaces

    def ufl_sub_spaces(self):
        return self._ufl_function_spaces

    def _ufl_hash_data_(self):
        return ("TensorProductFunctionSpace",) + tuple(V._ufl_hash_data_() for V in self.ufl_sub_spaces())

    def _ufl_signature_data_(self, renumbering):
        return ("TensorProductFunctionSpace",) + tuple(V._ufl_signature_data_(renumbering) for V in self.ufl_sub_spaces())

    def __repr__(self):
        r = "TensorProductFunctionSpace(*%s)" % repr(self._ufl_function_spaces)
        return as_native_str(r)
