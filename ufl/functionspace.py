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

# Export list for ufl.classes
__all_classes__ = [
    "AbstractFunctionSpace",
    "FunctionSpace",
    "MixedFunctionSpace",
    "TensorProductFunctionSpace",
    ]

class AbstractFunctionSpace(object):
    def ufl_sub_spaces(self):
        raise NotImplementedError("Missing implementation of IFunctionSpace.ufl_sub_spaces in %s." % self.__class__.__name__)

@attach_operators_from_hash_data
class FunctionSpace(AbstractFunctionSpace):
    # Slots are disabled here because they cause trouble in PyDOLFIN multiple inheritance pattern:
    #__slots__ = ("_ufl_domain", "_ufl_element")
    _ufl_noslots_ = True

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
        edata = repr(self.ufl_element())
        domain = self.ufl_domain()
        ddata = None if domain is None else domain._ufl_hash_data_()
        return ("FunctionSpace", ddata, edata)

    def _ufl_signature_data_(self, renumbering):
        edata = repr(self.ufl_element())
        domain = self.ufl_domain()
        ddata = None if domain is None else domain._ufl_signature_data_(renumbering)
        return ("FunctionSpace", ddata, edata)

    def __repr__(self):
        return "FunctionSpace(%r, %r)" % (self._ufl_domain, self._ufl_element)

@attach_operators_from_hash_data
class MixedFunctionSpace(AbstractFunctionSpace):
    # TODO: Disable slots also here? Will be subject to PyDOLFIN iheritance?
    __slots__ = ("_ufl_function_spaces",)
    def __init__(self, *function_spaces):
        AbstractFunctionSpace.__init__(self)
        self._ufl_function_spaces = function_spaces

    def ufl_sub_spaces(self):
        return (self._ufl_function_spaces,)

    def _ufl_hash_data_(self):
        return ("MixedFunctionSpace",) + tuple(V._ufl_hash_data_() for V in self.ufl_sub_spaces())

    def _ufl_signature_data_(self, renumbering):
        return ("MixedFunctionSpace",) + tuple(V._ufl_signature_data_(renumbering) for V in self.ufl_sub_spaces())

    def __repr__(self):
        return "MixedFunctionSpace(*%r)" % (self._ufl_function_spaces,)

@attach_operators_from_hash_data
class TensorProductFunctionSpace(AbstractFunctionSpace):
    # TODO: Disable slots also here? Will be subject to PyDOLFIN iheritance?
    __slots__ = ("_ufl_function_spaces",)
    def __init__(self, *function_spaces):
        AbstractFunctionSpace.__init__(self)
        self._ufl_function_spaces = function_spaces

    def ufl_sub_spaces(self):
        return (self._ufl_function_spaces,)

    def _ufl_hash_data_(self):
        return ("TensorProductFunctionSpace",) + tuple(V._ufl_hash_data_() for V in self.ufl_sub_spaces())

    def _ufl_signature_data_(self, renumbering):
        return ("TensorProductFunctionSpace",) + tuple(V._ufl_signature_data_(renumbering) for V in self.ufl_sub_spaces())

    def __repr__(self):
        return "TensorProductFunctionSpace(*%r)" % (self._ufl_function_spaces,)
