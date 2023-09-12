# -*- coding: utf-8 -*-
"Extract part of a form in a mixed FunctionSpace."

# Copyright (C) 2016 Chris Richardson and Lawrence Mitchell
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Cecile Daversin-Catty, 2018

from ufl.corealg.multifunction import reuse_if_untouched
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.constantvalue import Zero
from ufl.tensors import as_vector
from ufl.argument import Argument
from ufl.classes import Expr
from ufl.core.multiindex import MultiIndex
from ufl.functionspace import FunctionSpace
from functools import singledispatch


class FormSplitter(object):

    def __init__(self):
        self.cache = {}
        self.function = _form_splitter

    def __call__(self, node, *args):
        return self.function(node, self, *args)

    def split(self, form, ix, iy=0):
        # Remember which block to extract
        self.idx = [ix, iy]
        return map_integrand_dags(self, form)


@singledispatch
def _form_splitter(o, self, *args):
    """Single-dispatch function to split forms

    :arg o: UFL expression
    :arg self: wrapper class that managed which block we are extracting

    """
    raise AssertionError("UFL node expected, not %s" % type(o))


@_form_splitter.register(Expr)
def _form_splitter_expr(obj, self, *args):
    return reuse_if_untouched(obj, *args)


@_form_splitter.register(Argument)
def _form_splitter_argument(obj, self):
    if (obj.part() is not None):
        # Mixed element built from MixedFunctionSpace,
        # whose sub-function spaces are indexed by obj.part()
        if len(obj.ufl_shape) == 0:
            if (obj.part() == self.idx[obj.number()]):
                return obj
            else:
                return Zero()
        else:
            indices = [()]
            for m in obj.ufl_shape:
                indices = [(k + (j,)) for k in indices for j in range(m)]

            if (obj.part() == self.idx[obj.number()]):
                return as_vector([obj[j] for j in indices])
            else:
                return as_vector([Zero() for j in indices])
    else:
        # Mixed element built from MixedElement,
        # whose sub-elements need their function space to be created
        Q = obj.ufl_function_space()
        dom = Q.ufl_domain()
        sub_elements = obj.ufl_element().sub_elements()

        # If not a mixed element, do nothing
        if (len(sub_elements) == 0):
            return obj

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


@_form_splitter.register(MultiIndex)
def _form_spliter_multi_index(obj, self):
    return obj


def extract_blocks(form, i=None, j=None):
    fs = FormSplitter()
    arguments = form.arguments()
    forms = []

    numbers = tuple(sorted(set(a.number() for a in arguments)))
    arity = len(numbers)
    parts = tuple(sorted(set(a.part() for a in arguments)))
    assert arity <= 2

    if arity == 0:
        return (form, )

    for pi in parts:
        if arity > 1:
            for pj in parts:
                f = fs.split(form, pi, pj)
                if f.empty():
                    forms.append(None)
                else:
                    forms.append(f)
        else:
            f = fs.split(form, pi)
            if f.empty():
                forms.append(None)
            else:
                forms.append(f)

    try:
        forms_tuple = tuple(forms)
    except TypeError:
        # Only one form returned
        forms_tuple = (forms, )

    if i is not None:
        if arity > 1 and j is not None:
            return forms_tuple[i * len(parts) + j]
        else:
            return forms_tuple[i]
    else:
        return forms_tuple
