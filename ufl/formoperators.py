"Various high level ways to transform a complete Form into a new Form."

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-11-28"

from ufl.algorithms import compute_form_derivative, \
                           compute_form_adjoint, compute_form_action, \
                           compute_form_lhs, compute_form_rhs

# TODO: If we need to apply transformations inside the compiler,
# we must postphone the computations by using a framework like this.
# This could be important for AD.

class FormTransform(object):
    def __init__(self):
        pass

    def __add__(self, other):
        return FormTransformSum(self, other)

class FormTransformSum(FormTransform):
    def __init__(self, left, right):
        FormTransform.__init__(self)
        self._left = left
        self._right = right

class FormTransformDerivative(FormTransform):
    def __init__(self, form, function, basisfunction):
        FormTransform.__init__(self)
        self._form = form
        self._function = function
        self._basisfunction = basisfunction

class FormTransformAction(FormTransform):
    def __init__(self, form, function):
        FormTransform.__init__(self)
        self._form = form
        self._function = function

class FormTransformAdjoint(FormTransform):
    def __init__(self, form):
        FormTransform.__init__(self)
        self._form = form

class FormTransformLhs(FormTransform):
    def __init__(self, form):
        FormTransform.__init__(self)
        self._form = form

class FormTransformRhs(FormTransform):
    def __init__(self, form):
        FormTransform.__init__(self)
        self._form = form


def rhs(form):
    """Given a combined bilinear and linear form,
    extract the linear form part (right hand side).

    TODO: Given "a = u*v*dx + f*v*dx, should this
    return "+f*v*dx" as found in the form or
    "-f*v*dx" as the rigth hand side should
    be when solving the equations?
    """
    #return FormTransformRhs(form)
    return compute_form_rhs(form)

def lhs(form):
    """Given a combined bilinear and linear form,
    extract the bilinear form part (left hand side)."""
    #return FormTransformLhs(form)
    return compute_form_lhs(form)

def action(form, function=None):
    """Given a bilinear form, return a linear form
    with an additional function coefficient, representing
    the action of the form on the function. This can be
    used for matrix-free methods."""
    #return FormTransformAction(form)
    return compute_form_action(form, function)

def adjoint(form):
    """Given a combined bilinear form, compute the adjoint
    form by swapping the test and trial functions."""
    #return FormTransformAdjoint(form)
    return compute_form_adjoint(form)

def derivative(form, function, basisfunction=None):
    """Given any form, compute the linearization of the
    form with respect to the given discrete function.
    The resulting form has one additional basis function
    in the same finite element space as the function.
    A tuple of Functions may be provided in place of
    a single Function, in which case the new BasisFunction
    argument is based on a MixedElement created from this tuple."""
    #return FormTransformDerivative(form)
    return compute_form_derivative(form, function, basisfunction)

