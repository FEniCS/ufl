from ufl.utils.str import as_native_str
from ufl.core.ufl_type import ufl_type
from ufl.core.terminal import Terminal
from ufl.domain import as_domain
from ufl.utils.counted import counted_init


@ufl_type()
class Constant(Terminal):
    """UFL Constant"""
    _ufl_noslots_ = True
    _globalcount = 0

    def __init__(self, domain, shape=(), count=None):
        Terminal.__init__(self)
        counted_init(self, count=count, countedclass=Constant)

        self._ufl_domain = as_domain(domain)
        self._ufl_shape = shape

        self._repr = as_native_str("Constant(%s, %s, %s)" % (
            repr(self._ufl_domain), repr(self._ufl_shape), repr(self._count)))

    def count(self):
        return self._count

    @property
    def ufl_shape(self):
        "Return the associated UFL shape."
        return self._ufl_shape

    def ufl_domain(self):
        return self._ufl_domain

    def ufl_domains(self):
        return (self.ufl_domain(), )

    def is_cellwise_constant(self):
        return True

    def __str__(self):
        count = str(self._count)
        return "c_{%s}" % count

    def __repr__(self):
        return self._repr


def VectorConstant(domain, count=None):
    domain = as_domain(domain)
    return Constant(domain, shape=(domain.geometric_dimension(), ), count=count)


def TensorConstant(domain, count=None):
    domain = as_domain(domain)
    return Constant(domain, shape=(domain.geometric_dimension(), domain.geometric_dimension()), count=count)
