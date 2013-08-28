
from ufl.operatorbase import Operator
from ufl.terminal import Terminal
from ufl.common import fast_pre_traversal

def _expr_equals0(a, b): # TODO: Which is faster?
    # Cutoff for different type
    if type(a) != type(b):
        return False

    # Cutoff for same object
    if a is b:
        return True

    # Iterate over pairs of potentially matching subexpressions
    input = [(a, b)]
    while input:
        a, b = input.pop()

        # Cutoff for different type
        if type(a) != type(b):
            return False

        # Get operands
        aops = a.operands()
        bops = b.operands()
        if aops:
            if len(aops) != len(bops):
                return False
            # Add children for checking
            input.extend(izip(aops, bops))
        else:
            # Compare terminals
            if not a == b:
                return False

    # Everything checked out fine, expressions must be equal
    return True

def _expr_equals1(a, b): # TODO: Which is faster?
    # Cutoff for different type
    if type(a) != type(b):
        return False
    # Cutoff for same object
    if a is b:
        return True
    # Compare entire expression structure
    for x,y in izip(fast_pre_traversal(a), fast_pre_traversal(b)):
        if type(x) != type(y):
            return False
        #if isinstance(Terminal, x) and not x == y:
        if x.operands() == () and not x == y:
            return False
    # Equal terminals and types, a and b must be equal
    return True

def _expr_equals2(a, b):
    # Cutoff for different type
    if type(a) != type(b):
        return False
    # Cutoff for same object
    if a is b:
        return True
    from ufl.algorithms.traversal import traverse_terminals, traverse_operands
    # Check for equal terminals
    for x,y in izip(traverse_terminals(a), traverse_terminals(b)):
        if x != y:
            return False
    # Check for matching operator types
    for x,y in izip(traverse_operands(a), traverse_operands(b)):
        if type(x) != type(y):
            return False
    # Equal terminals and operands, a and b must be equal
    return True

equalsrecursed = {}
equalscalls = {}
collisions = {}
def print_collisions():
    print
    print "Collision statistics:"
    keys = sorted(equalscalls.keys(), key=lambda x: collisions.get(x,0))
    for k in keys:
        co = collisions.get(k,0)
        ca = equalscalls[k]
        print k, co, ca, int(100.0*co/ca)
    print "Recursion statistics:"
    keys = sorted(keys, key=lambda x: equalsrecursed.get(x,0))
    for k in keys:
        r = equalsrecursed.get(k,0)
        ca = equalscalls[k]
        print k, r, ca, int(100.0*r/ca)
    print

def _expr_equals3(self, other): # Much faster than the more complex algorithms above!
    """Checks whether the two expressions are represented the
    exact same way. This does not check if the expressions are
    mathematically equal or equivalent! Used by sets and dicts."""

    # Code for counting number of equals calls:
    #equalscalls[type(self)] = equalscalls.get(type(self),0) + 1

    # Fast cutoff for common case
    if type(self) != type(other):
        return False

    # TODO: Test how this affects the run time:
    # Compare hashes if hash is cached
    # (NB! never access _hash directly, it may be computed on demand in __hash__)
    if (hasattr(self, "_hash") and hash(self) != hash(other)):
        return False

    # Large objects are costly to compare with themselves
    if self is other:
        return True

    if isinstance(self, Operator):
        # Just let python handle the recursion
        #equal = self.operands() == other.operands()
        # Recurse manually to call _expr_equals3 directly without the class EQ overhead!
        equal = all(_expr_equals3(a, b) for (a,b) in zip(self.operands(), other.operands()))
    else:
        # Compare terminal representations to include all terminal data
        #equal = repr(self) == repr(other)
        # Compare terminals with custom == to capture subclass overloading of __eq__
        equal = self == other

    # At this point, self and other has the same hash, and equal _should_ be True...
    # Code for measuring amount of collisions:
    #if not equal:
    #    collisions[type(self)] = collisions.get(type(self), 0) + 1

    # Code for counting number of recursive calls:
    #equalsrecursed[type(self)] = equalsrecursed.get(type(self),0) + 1

    # Debugging check: (has been enabled for a long while without any fails as of nov. 30th 2012
    #req = repr(self) == repr(other)
    #if req != equal: # This may legally fail for test/trial functions from PyDOLFIN
    #    print '\n'*3
    #    print self
    #    print other
    #    print '\n'*3
    #    ufl_error("INVALID COMPARISON!")

    return equal

expr_equals = _expr_equals3
