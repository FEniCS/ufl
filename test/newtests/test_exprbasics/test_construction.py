
# Create cFoo functions for all expression hierarchy classes
from ufl.classes import all_ufl_classes
for c in all_ufl_classes:
    exec "%s = c" % c.__name__
    exec "def c%s(*args, **kwargs): return %s(*args, **kwargs)" % (c.__name__, c.__name__)

# TODO: Need to define valid argument groups for each type of expression in some way
def _test_creation():
    for c in all_ufl_classes:
        cf = globals()["c%s" % c.__name__]
        yield cf, 1, 2, 3

# TODO: Define argument ranges for all literals and geometric quantities, this can easily be done manually
from ufl.classes import IntValue, FloatValue, Identity
from ufl import as_ufl
def test_literals():
    for i in range(3):
        assert IntValue(i) == as_ufl(i)
    for i in range(3):
        assert FloatValue(0.1*i) == as_ufl(0.1*i)
    for i in range(1,3):
        assert Identity(i).shape() == (i, i)

