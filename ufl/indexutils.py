
#--- Utility functions ---

#       # Extract repeated indices (no context here, so we're not judging what's valid)
#       ri = []
#       count = {}
#       for i in self._indices:
#           if i in count:
#               count[i] += 1
#               ri.append(i)
#           else:
#               count[i] = 1
#       if any(c > 2 for c in count.values()):
#           warning("Found more than two repeated indices in MultiIndex, this will likely fail later?")
#       self._repeated_indices = tuple(ri)

def complete_shape(shape, default_dim):
    "Complete shape tuple by replacing non-integers with a default dimension."
    return tuple((s if isinstance(s, int) else default_dim) for s in shape)

def get_common_indices(a, b): # Not in use
    ai = a.free_indices()
    bi = b.free_indices()
    cis = set(ai) ^ set(bi)
    return cis

def get_free_indices(a, b): # Not in use
    ai = a.free_indices()
    bi = b.free_indices()
    cis = set(ai) ^ set(bi)
    return tuple(i for i in chain(ai,bi) if not i in cis)

def split_indices(a, b): # Not in use
    ai = a.free_indices()
    bi = b.free_indices()
    ais = set(ai)
    bis = set(bi)
    ris = ais ^ bis
    fi  = tuple(i for i in chain(ai,bi) if not i in ris)
    ri  = tuple(i for i in chain(ai,bi) if     i in ris)
    #n = len(ri) + len(fi)
    #ufl_assert(n == ?)
    return (fi, ri)

