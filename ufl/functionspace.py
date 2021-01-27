# -*- coding: utf-8 -*-
"Types for representing function spaces."

# Copyright (C) 2015-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Massimiliano Leoni, 2016
# Modified by Cecile Daversin-Catty, 2018

from ufl.log import error
from ufl.core.ufl_type import attach_operators_from_hash_data
from ufl.domain import join_domains

# Export list for ufl.classes
__all_classes__ = [
    "AbstractFunctionSpace",
    "FunctionSpace",
    "DualSpace",
    "MixedFunctionSpace",
    "TensorProductFunctionSpace",
]


class AbstractFunctionSpace(object):
    def ufl_sub_spaces(self):
        raise NotImplementedError("Missing implementation of IFunctionSpace.ufl_sub_spaces in %s." % self.__class__.__name__)


@attach_operators_from_hash_data
class BaseFunctionSpace(AbstractFunctionSpace):
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
        return r


class FunctionSpace(BaseFunctionSpace):
    def __init__(self, domain, element):
        BaseFunctionSpace.__init__(self, domain, element)

    def dual(self):
        return DualSpace(self._ufl_domain, self._ufl_element)


class DualSpace(BaseFunctionSpace):
    def __init__(self, domain, element):
        BaseFunctionSpace.__init__(self, domain, element)

    def dual(self):
        return FunctionSpace(self._ufl_domain, self._ufl_element)


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
        return r


@attach_operators_from_hash_data
class MixedFunctionSpace(AbstractFunctionSpace):
    def __init__(self, *args):
        AbstractFunctionSpace.__init__(self)
        self._ufl_function_spaces = args
        self._ufl_elements = list()
        for fs in args:
            if isinstance(fs, BaseFunctionSpace):
                self._ufl_elements.append(fs.ufl_element())
            else:
                error("Expecting BaseFunctionSpace objects")

    def ufl_sub_spaces(self):
        "Return ufl sub spaces."
        return self._ufl_function_spaces

    def ufl_sub_space(self, i):
        "Return i-th ufl sub space."
        return self._ufl_function_spaces[i]

    def ufl_dual_space(self, *args):
        if args:
            spaces = [0 for i in range(len(self._ufl_function_spaces))]
            for i in range(len(self._ufl_function_spaces)):
                if i in args:
                    spaces[i] = self._ufl_function_spaces[i].ufl_dual_space()
                else:
                    spaces[i] = self._ufl_function_spaces[i]
            return MixedFunctionSpace(*spaces)
        else:
            return MixedFunctionSpace(*[space.ufl_dual_space() for space in self._ufl_function_spaces])

    def ufl_elements(self):
        "Return ufl elements."
        return self._ufl_elements

    def ufl_element(self):
        if len(self._ufl_elements) == 1:
            return self._ufl_elements[0]
        else:
            error("""Found multiple elements. Cannot return only one.
            Consider building a FunctionSpace from a MixedElement
            in case of homogeneous dimension.""")

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

    def num_sub_spaces(self):
        return len(self._ufl_function_spaces)

    def _ufl_hash_data_(self):
        return ("MixedFunctionSpace",) + tuple(V._ufl_hash_data_() for V in self.ufl_sub_spaces())

    def _ufl_signature_data_(self, renumbering):
        return ("MixedFunctionSpace",) + tuple(V._ufl_signature_data_(renumbering) for V in self.ufl_sub_spaces())

    def __repr__(self):
        r = "MixedFunctionSpace(*%s)" % repr(self._ufl_function_spaces)
        return r
