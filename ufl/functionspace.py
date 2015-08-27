# -*- coding: utf-8 -*-
"Types for representing function spaces."

# Copyright (C) 2015-2015 Martin Sandve Alnæs
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

#from ufl.core.terminal import Terminal
#from ufl.corealg.traversal import traverse_unique_terminals
#from ufl.utils.formatting import istr
#from ufl.utils.dicts import EmptyDict
#from ufl.log import warning, error, deprecate
#from ufl.assertions import ufl_assert
#from ufl.protocols import id_or_none
#from ufl.cell import as_cell, AbstractCell, Cell, ProductCell

from ufl.core.ufl_type import attach_operators_from_hash_data

class AbstractFunctionSpace(object):
    def ufl_sub_spaces(self):
        raise NotImplementedError("Missing implementation of IFunctionSpace.ufl_sub_spaces in %s." % self.__class__.__name__)

@attach_operators_from_hash_data
class FunctionSpace(AbstractFunctionSpace):
    __slots__ = ("_ufl_domain", "_ufl_element")

    def __init__(self, domain, element):
        AbstractFunctionSpace.__init__(self)
        self._ufl_domain = domain
        self._ufl_element = element

    def ufl_sub_spaces(self):
        return ()

    def ufl_domain(self):
        return self._ufl_domain

    def ufl_element(self):
        return self._ufl_element

    def _ufl_hash_data_(self):
        return (self.__class__.__name__,) + (self.ufl_domain()._ufl_hash_data_(), self.ufl_element()._ufl_hash_data_())

    def _ufl_signature_data_(self, renumbering):
        return (self.ufl_domain()._ufl_signature_data_(renumbering),
                self.ufl_element()._ufl_signature_data_(renumbering))

@attach_operators_from_hash_data
class MixedFunctionSpace(AbstractFunctionSpace):
    __slots__ = ("_ufl_function_spaces",)
    def __init__(self, function_spaces):
        AbstractFunctionSpace.__init__(self)
        self._ufl_function_spaces = function_spaces

    def ufl_sub_spaces(self):
        return (self._ufl_function_spaces,)

    def _ufl_hash_data_(self):
        return (self.__class__.__name__,) + tuple(V._ufl_hash_data_() for V in self.ufl_sub_spaces())

    def _ufl_signature_data_(self, renumbering):
        return tuple(V._ufl_signature_data_(renumbering) for V in self.ufl_sub_spaces())

@attach_operators_from_hash_data
class TensorProductFunctionSpace(AbstractFunctionSpace):
    __slots__ = ("_ufl_function_spaces",)
    def __init__(self, function_spaces):
        AbstractFunctionSpace.__init__(self)
        self._ufl_function_spaces = function_spaces

    def ufl_sub_spaces(self):
        return (self._ufl_function_spaces,)

    def _ufl_hash_data_(self):
        return (self.__class__.__name__,) + tuple(V._ufl_hash_data_() for V in self.ufl_sub_spaces())

    def _ufl_signature_data_(self, renumbering):
        return tuple(V._ufl_signature_data_(renumbering) for V in self.ufl_sub_spaces())
