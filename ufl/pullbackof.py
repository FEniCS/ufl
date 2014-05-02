
# FIXME: Finish the implementation of this type.

from ufl.operator import Operator
from ufl.common import EmptyDict

class PullbackOf(Operator):
    "Pullback of form argument to reference cell representation."
    __slots__ = ("_op",)
    def __init__(self, f):
        Operator.__init__(self)
        self._f = f
        error("Not fully implemented.")

    def operands(self):
        return (self._f,)

    def free_indices(self):
        return ()

    def index_dimensions(self):
        return EmptyDict

    def shape(self):
        return self._f.element().reference_value_shape()
