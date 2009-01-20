
# FIXME: Decide on this thingy. If we decide to go for it, here's the plan:
#
# --- Step A: Doing no harm, can be done before final decision ---
#
# - Implement self.new_free_indices() in Product to return all unique indices.
# - Implement self.new_free_indices() in Indexed to return all unique indices.
# - Implement self.new_free_indices() in SpatialDerivative to return all unique indices.
#
# At this point, reconsider 
#
# --- Step B: Core modifications ---
#
# - Update __mul__ to use IndexSum
# - Update Product: remove repeated_indices() and free_indices(), rename new_free_indices to free_indices
#
# - Update __getitem__ to use IndexSum
# - Update Indexed: remove repeated_indices() and free_indices(), rename new_free_indices to free_indices
#
# - Update .dx to use IndexSum
# - Update SpatialDerivative: remove repeated_indices() and free_indices(), rename new_free_indices to free_indices
#
# --- Step C: Update all small stuff ---
#
# - Remove repeated_indices() from Expr and everywhere else.
# - Step through the rest of the free_indices() implementations. Most if not all should just be the same as before.
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


