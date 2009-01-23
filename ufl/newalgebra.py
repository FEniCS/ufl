
#
# FIXME: Decide on this thingy. If we decide to go for it, here's the plan:
#
# --- Step A: Doing no harm, can be done before final decision ---
#
# - Implement self.new_free_indices() in Product to return all unique indices.
# - Implement self.new_free_indices() in Indexed to return all unique indices.
# - Implement self.new_free_indices() in SpatialDerivative to return all unique indices.
# - Implement new __mul__ to use IndexSum
# - Implement new __getitem__ to use IndexSum
# - Implement new .dx to use IndexSum
# - Step through the rest of the free_indices() implementations. Most if not all should just be the same as before, make new_free_indices where needed.
#
# --- Step B: Set modifications to use ---
#
# - Remove repeated_indices() everywhere
# - Rename new_free_indices to free_indices() everywhere needed
# - Replace old __mul__, __getitem__, .dx with new ones
#
# Test!
#
# --- Step D: Update algorithms ---
#
# - Get overview of algorithms, consider the following new issues:
#   - repeated_indices is gone
#   - no implicit sums
#   - Must add indexed_sum support
#   - Dot(a, b) -> a[i]*b[i] will still work through __mul__
#   - Dot(a, b) -> Product(a[i], b[i]) will be wrong!
#
# - Write up all algorithms here with notes:
#   - TODO: Need a more thourough plan for this.
#
# - Expr.evaluate  (depends on expand_compounds and expand_derivatives)
#


# --- New:

class IndexSum(Expr):
    __slots__ = ("_summand", "_index", "_repr")
    
    def __init__(self, summand, index):
        Expr.__init__(self)
        ufl_assert(isinstance(summand, Expr), "Expecting Expr instances.")
        if isinstance(index, Index):
            index = MultiIndex((index,))
        ufl_assert(isinstance(index, MultiIndex), "Expecting (Multi)Index instance.")
        ufl_assert(len(index) == 1, "Expecting single Index.")
        self._summand = summand
        self._index = index
        self._repr = "IndexSum(%r, %r)" % (summand, index)
    
    def operands(self):
        return (self._summand, self._index)
    
    def indices(self):
        j = self._index[0]
        return tuple(i for i in self._summand.free_indices() if not i == j)
    
    def index_dimensions(self):
        return self._operands[0].index_dimensions()
    
    def shape(self):
        return self._summand.shape()
    
    def evaluate(self, x, mapping, component, index_values):
        return sum(o.evaluate(x, mapping, component, index_values) for o in self.operands())
    
    def __str__(self):
        return "sum_{%s}< %s >" % (str(self._index), str(self._summand))
    
    def __repr__(self):
        return self._repr

# --- In indexing.py:

def build_unique_indices(operands):
    "Build tuple of unique indices, including repeated ones."
    s = set()
    fi = []
    idims = {}
    for o in operands:
        if isinstance(o, MultiIndex):
            ofi = o._indices
            oid = dict((i, None) for i in ofi) # TODO: This introduces None, better way?
        else:
            ofi = o.free_indices()
            oid = o.index_dimensions()
        for i in ofi:
            if i in s:
                ri.append(i)
            else:
                fi.append(i)
                idims[i] = oid[i]
                s.add(i)
    return fi, ri, idims

class MultiIndex(Expr):
    def __init__(self, ii):
        fi = set(ii)
        self._fi = fi
    
    def free_indices(self):
        return self._fi

class Indexed(Expr):
    def __init__(self, A, ii):
        fi, ri, idims = build_unique_indices((A, ii))
        self._fi = fi
        self._idims = idims
    
    def free_indices(self):
        return self._fi
    
    def index_dimensions(self):
        return self._idims

# --- In algebra.py:

class Product(Expr):
    def __init__(self, *operands):
        fi, ri, idims = build_unique_indices(operands)
        self._fi = fi
        self._idims = idims
    
    def free_indices(self):
        return self._fi
    
    def index_dimensions(self):
        return self._idims

# --- In differentiation.py:

class SpatialDerivative(Expr):
    def __init__(self, f, ii):
        fi, ri, idims = build_unique_indices((f, ii))
        self._fi = fi
        self._idims = idims
    
    def free_indices(self):
        return self._fi
    
    def index_dimensions(self):
        return self._idims

# --- In exproperators.py:

def _mult(a, b): # TODO: Rewrite
    
    fi, ri, idims = build_unique_indices((a, b))
    
    # Pick out valid non-scalar products here (dot products):
    # - matrix-matrix (A*B, M*grad(u)) => A . B
    # - matrix-vector (A*v) => A . v
    s1 = a.shape()
    s2 = b.shape()
    l1 = len(s1)
    l2 = len(s2)
    if l1 == 2 and (l2 == 2 or l2 == 1):
        ufl_assert(not ri, "Not expecting repeated indices in non-scalar product.")
        shape = s1[:-1] + s2[1:]
        if isinstance(a, Zero) or isinstance(b, Zero):
            return Zero(shape, fi, idims)
        i = Index()
        return a[...,i]*b[i,...] # TODO: Does [...,i] work with vectors?
    
    # Scalar products use Product and IndexSum for implicit sums:
    p = Product(a, b)
    for i in ri:
        p = IndexSum(p, i)
    return p

def _dx(self, *ii): # TODO: Rewrite
    "Return the partial derivative with respect to spatial variable number i."
    fi, ri, idims = build_unique_indices((ii,))
    d = self
    # Apply all derivatives
    for i in ii:
        d = SpatialDerivative(d, i)
    # Apply all implicit sums
    for i in ri:
        d = IndexSum(d, i)
    return d
Expr.dx = _dx

def _d(self, v): # TODO: Rewrite
    "Return the partial derivative with respect to variable v."
    # TODO: Maybe v can be an Indexed of a Variable, in which case we can use indexing to extract the right component?
    return VariableDerivative(self, v)
Expr.d = _d



# TODO: Check this out:
class ComponentTensor(Expr):
    __slots__ = ("_expression", "_indices", "_free_indices", "_index_dimensions", "_shape")
    
    def __init__(self, expression, indices):
        Expr.__init__(self)
        ufl_assert(isinstance(expression, Expr), "Expecting ufl expression.")
        ufl_assert(expression.shape() == (), "Expecting scalar valued expression.")
        self._expression = expression
        
        ufl_assert(all(isinstance(i, Index) for i in indices),
           "Expecting sequence of Index objects, not %s." % repr(indices))
        
        if not isinstance(indices, MultiIndex): # if constructed from repr
            indices = MultiIndex(indices)
        self._indices = indices
        
        eset = set(expression.free_indices())
        iset = set(self._indices)
        freeset = eset - iset
        missingset = iset - eset
        self._free_indices = tuple(freeset)
        ufl_assert(len(missingset) == 0, "Missing indices %s in expression %s." % (missingset, expression))
        
        dims = expression.index_dimensions()
        self._index_dimensions = dict((i, dims[i]) for i in self._free_indices)
        
        self._shape = tuple(dims[i] for i in self._indices)
    
    def operands(self):
        return (self._expression, self._indices)
    
    def free_indices(self):
        return self._free_indices
    
    def index_dimensions(self):
        return self._index_dimensions
    
    def shape(self):
        return self._shape
    
    def evaluate(self, x, mapping, component, index_values):
        indices = self._indices
        a = self._expression
        
        for i, c in enumerate(indices, component):
            index_values.push(i, c)
        
        a = a.evaluate(x, mapping, (), index_values)
        
        for i in indices:
            index_values.pop()
        
        return a
    
    def __str__(self):
        return "[Rank %d tensor A, such that A_{%s} = %s]" % (self.rank(), self._indices, self._expression)
    
    def __repr__(self):
        return "ComponentTensor(%r, %r)" % (self._expression, self._indices)
