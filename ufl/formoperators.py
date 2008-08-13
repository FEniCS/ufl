"""Various high level ways to transform a complete Form."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-05-20"


# TODO: We should be able to apply most of these operators to parts of a form,
#       so should they be applied to integrand expressions and be UFLObjects?


# TODO: We should be able to take the derivative of a part of the form, doing some linearization manually.
class Derivative:
    """..."""
    def __init__(self, form, function):
        self.form = form
        self.function = function
    
    def __repr__(self):
        return "Derivative(%r, %r)" % (self.form, self.function)


# Deprecated by the more general Derivative, or should we keep it for the familiar name?
#class Jacobi:
#    """Represents a linearized form, the Jacobi of a given nonlinear form wrt a given function."""
#    def __init__(self, form, function):
#        self.form = form
#        self.function = function
#
#    def __repr__(self):
#        return "Jacobi(%s, %s)" % (repr(self.form), repr(self.function))


class Action:
    """Represents the action of a linear form of rank 2 on a vector."""
    def __init__(self, form):
        self.form = form

    def __repr__(self):
        return "Action(%r)" % self.form


class Rhs:
    """Represents the right hand side part of a form, that is the rank 1 part."""
    def __init__(self, form):
        self.form = form

    def __repr__(self):
        return "Rhs(%r)" % self.form


class Lhs:
    """Represents the left hand side part of a form, that is the rank 2 part."""
    def __init__(self, form):
        self.form = form

    def __repr__(self):
        return "Lhs(%r)" % self.form


def rhs(form):
    return Rhs(form)

def lhs(form):
    return Lhs(form)
