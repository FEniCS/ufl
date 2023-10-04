"""Expr equals."""

from collections import defaultdict

hash_total = defaultdict(int)
hash_collisions = defaultdict(int)
hash_equals = defaultdict(int)
hash_notequals = defaultdict(int)


def expr_equals(self, other):
    """Checks whether the two expressions are represented the exact same way.

    This does not check if the expressions are
    mathematically equal or equivalent! Used by sets and dicts.
    """
    # Fast cutoffs for common cases, type difference or hash
    # difference will cutoff more or less all nonequal types
    if type(self) is not type(other) or hash(self) != hash(other):
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
