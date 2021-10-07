# -*- coding: utf-8 -*-

from collections import defaultdict

from ufl.core.expr import Expr
from ufl.log import error

hash_total = defaultdict(int)
hash_collisions = defaultdict(int)
hash_equals = defaultdict(int)
hash_notequals = defaultdict(int)


def print_collisions():

    keys = sorted(hash_total.keys(), key=lambda x: (hash_collisions[x], x))

    print("Collision statistics ({0} keys):".format(len(keys)))
    print("[key: equals; notequals; collisions]")
    n = max(len(str(k)) for k in keys)
    fmt = ("%%%ds" % n) + ": \t %6d (%3d%%); %6d (%3d%%); %6d (%3d%%) col; tot %d"
    for k in keys:
        co = hash_collisions[k]
        eq = hash_equals[k]
        ne = hash_notequals[k]
        tot = hash_total[k]
        sn, on = k
        # Skip those that are all not equal
        if sn != on and ne == tot:
            continue
        print(fmt % (k, eq, int(100.0 * eq / tot),
                     ne, int(100.0 * ne / tot),
                     co, int(100.0 * co / tot),
                     tot))


def measure_collisions(equals_func):
    def equals_func_with_collision_measuring(self, other):
        # Call equals
        equal = equals_func(self, other)

        # Get properties
        st = type(self)
        ot = type(other)
        sn = st.__name__
        on = ot.__name__
        sh = hash(self)
        oh = hash(other)
        key = (sn, on)

        # If hashes are the same but objects are not equal, we have a
        # collision
        hash_total[key] += 1
        if sh == oh and not equal:
            hash_collisions[key] += 1
        elif sh != oh and equal:
            error("Equal objects must always have the same hash! Objects are:\n{0}\n{1}".format(self, other))
        elif sh == oh and equal:
            hash_equals[key] += 1
        elif sh != oh and not equal:
            hash_notequals[key] += 1

        return equal
    return equals_func_with_collision_measuring


# @measure_collisions
def recursive_expr_equals(self, other):  # Much faster than the more complex algorithms above!
    """Checks whether the two expressions are represented the
    exact same way. This does not check if the expressions are
    mathematically equal or equivalent! Used by sets and dicts."""

    # To handle expr == int/float
    if not isinstance(other, Expr):
        return False

    # Fast cutoff for common case
    if self._ufl_typecode_ != other._ufl_typecode_:
        return False

    # Compare hashes, will cutoff more or less all nonequal types
    if hash(self) != hash(other):
        return False

    # Large objects are costly to compare with themselves
    if self is other:
        return True

    # Terminals
    if self._ufl_is_terminal_:
        # Compare terminals with custom == to capture subclass
        # overloading of __eq__
        return self == other

    # --- Operators, most likely equal, below here is the costly part
    # --- if it recurses through a large tree! ---

    # Recurse manually to call expr_equals directly without the class
    # EQ overhead!
    equal = all(recursive_expr_equals(a, b) for (a, b) in zip(self.ufl_operands,
                                                              other.ufl_operands))

    return equal


# @measure_collisions
def nonrecursive_expr_equals(self, other):
    """Checks whether the two expressions are represented the
    exact same way. This does not check if the expressions are
    mathematically equal or equivalent! Used by sets and dicts."""

    # Fast cutoffs for common cases, type difference or hash
    # difference will cutoff more or less all nonequal types
    if type(self) != type(other) or hash(self) != hash(other):
        return False

    # Large objects are costly to compare with themselves
    if self is other:
        return True

    # Modelled after pre_traversal to avoid recursion:
    left = [(self, other)]
    while left:
        s, o = left.pop()

        if s._ufl_is_terminal_:
            # Compare terminals
            if not s == o:
                return False
        else:
            # Delve into subtrees
            so = s.ufl_operands
            oo = o.ufl_operands
            if len(so) != len(oo):
                return False

            for s, o in zip(so, oo):
                # Fast cutoff for common case
                if s._ufl_typecode_ != o._ufl_typecode_:
                    return False
                # Skip subtree if objects are the same
                if s is o:
                    continue
                # Append subtree for further inspection
                left.append((s, o))

    # Equal if we get out of the above loop!
    # Eagerly DAGify to reduce the size of the tree.
    self.ufl_operands = other.ufl_operands
    return True


# expr_equals = recursive_expr_equals
expr_equals = nonrecursive_expr_equals
