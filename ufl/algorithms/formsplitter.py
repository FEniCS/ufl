# -*- coding: utf-8 -*-
"Extract part of a form in a mixed FunctionSpace."

# Copyright (C) 2016 Chris Richardson and Lawrence Mitchell
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

from ufl.corealg.multifunction import MultiFunction
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.constantvalue import Zero
from ufl.tensors import as_vector
from ufl.argument import Argument
from ufl.functionspace import FunctionSpace, FunctionSpaceProduct


class FormSplitter(MultiFunction):

    def split(self, form, ix, iy=0):
        # Remember which block to extract
        self.idx = [ix, iy]
        return map_integrand_dags(self, form)

    def argument(self, obj):
        Q = obj.ufl_function_space()
        dom = Q.ufl_domain()
        sub_elements = obj.ufl_element().sub_elements()

        # If not a mixed element, do nothing
        if (len(sub_elements) == 0):
            return obj

        # Split into sub-elements, creating appropriate space for each
        args = []
        for i, sub_elem in enumerate(sub_elements):
            Q_i = FunctionSpace(dom, sub_elem)
            a = Argument(Q_i, obj.number(), part=obj.part())

            indices = [()]
            for m in a.ufl_shape:
                indices = [(k + (j,)) for k in indices for j in range(m)]

            if (i == self.idx[obj.number()]):
                args += [a[j] for j in indices]
            else:
                args += [Zero() for j in indices]

        return as_vector(args)

    def multi_index(self, obj):
        return obj

    expr = MultiFunction.reuse_if_untouched

def block_split(form, ix, iy=0):
    fs = FormSplitter()
    return fs.split(form, ix, iy)

class FormSplitterProduct(MultiFunction):

    def split(self, form, i, j=0):
        self.idx = [i,j]
        return map_integrand_dags(self, form)

    def argument(self, obj):
        # obj.part correspond to the subdomain index here
        if (obj.part() == self.idx[obj.number()]):
            if len(obj.ufl_shape) == 0:
                return obj
            else:
                error("TMP: ufl_shape should be zero")
        else:
            return Zero()

    def multi_index(self, obj):
        return obj

    expr = MultiFunction.reuse_if_untouched
