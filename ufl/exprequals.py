"""Expr equals."""


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
    if (self is other) or (self.ufl_operands is other.ufl_operands):
        return True

    # Modelled after pre_traversal to avoid recursion:
    left = [(self, other)]
    equal_pairs = set()
    while left:
        pair = left.pop()
        s, o = pair
        if s._ufl_is_terminal_:
            # Compare terminals
            if not s == o:
                return False
        else:
            # Delve into subtrees
            so = s.ufl_operands
            oo = o.ufl_operands
            # Skip subtree if operands are the same
            if so is oo:
                continue
            if len(so) != len(oo):
                return False

            for s, o in zip(so, oo):
                # Fast cutoff for common case
                if s._ufl_typecode_ != o._ufl_typecode_:
                    return False
                # Skip subtree if objects are the same
                if s is o:
                    continue
                if (id(s), id(o)) in equal_pairs:
                    continue
                # Append subtree for further inspection
                left.append((s, o))

        # Keep track of equal subexpressions
        equal_pairs.add((id(pair[0]), id(pair[1])))

    # Equal if we get out of the above loop!
    # Eagerly DAGify to reduce the size of the tree.
    self.ufl_operands = other.ufl_operands
    return True
