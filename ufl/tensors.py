"""Classes used to group scalar expressions into expressions with rank > 0."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-31 -- 2009-02-20"

from ufl.log import warning
from ufl.assertions import ufl_assert
from ufl.expr import Expr, WrapperType
from ufl.constantvalue import as_ufl
from ufl.indexing import Index, FixedIndex, MultiIndex, indices

# --- Classes representing tensors of UFL expressions ---

class ListTensor(WrapperType):
    __slots__ = ("_expressions", "_free_indices", "_shape", "_repr")
    
    def __init__(self, *expressions):
        WrapperType.__init__(self)
        if isinstance(expressions[0], (list, tuple)):
            expressions = [ListTensor(*sub) for sub in expressions]
        
        if not all(isinstance(e, ListTensor) for e in expressions):
            expressions = [as_ufl(e) for e in expressions]
            ufl_assert(all(isinstance(e, Expr) for e in expressions), \
                "Expecting list of subtensors or expressions.")
        
        self._expressions = tuple(expressions)
        r = len(expressions)
        e = expressions[0]
        c = e.shape()
        self._shape = (r,) + c
        
        ufl_assert(all(sub.shape() == c for sub in expressions),
            "Inconsistent subtensor size.")
        
        indexset = set(e.free_indices())
        ufl_assert(all(not (indexset ^ set(sub.free_indices())) for sub in expressions), \
            "Can't combine subtensor expressions with different sets of free indices.")

        self._repr = "ListTensor(%s)" % ", ".join(repr(e) for e in self._expressions)
    
    def operands(self):
        return self._expressions
    
    def free_indices(self):
        return self._expressions[0].free_indices()
    
    def index_dimensions(self):
        return self._expressions[0].index_dimensions()
    
    def shape(self):
        return self._shape
    
    def evaluate(self, x, mapping, component, index_values):
        a = self._expressions[component[0]]
        component = component[1:]
        a = a.evaluate(x, mapping, component, index_values)
        return a
    
    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        k = key[0]
        if isinstance(k, (int, FixedIndex)):
            sub = self._expressions[int(k)]
            return sub if len(key) == 1 else sub[key[1:]]
        return Expr.__getitem__(self, key)
    
    def __str__(self):
        def substring(expressions, indent):
            ind = " "*indent
            if isinstance(expressions[0], ListTensor):
                s = (ind+",\n").join(substring(e._expressions, indent+2) for e in expressions)
                return ind + "[" + "\n" + s + "\n" + ind + "]"
            else:
                return ind + "[ %s ]" % ", ".join(str(e) for e in expressions)
        sub = substring(self._expressions, 0)
        return "[%s]" % sub
    
    def __repr__(self):
        return self._repr

class ComponentTensor(WrapperType):
    __slots__ = ("_expression", "_indices", "_free_indices", "_index_dimensions", "_shape", "_str", "_repr")
    
    def __init__(self, expression, indices):
        WrapperType.__init__(self)
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

        self._str = "< Rank %d tensor A, such that A_{%s} = %s >" % (self.rank(), self._indices, self._expression)
        self._repr = "ComponentTensor(%r, %r)" % (self._expression, self._indices)
    
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
        
        # Map component to indices
        for i, c in zip(indices, component):
            index_values.push(i, c)
        
        a = a.evaluate(x, mapping, (), index_values)
        
        for _ in component:
            index_values.pop()
        
        return a
    
    def __str__(self):
        return self._str
    
    def __repr__(self):
        return self._repr

# --- User-level functions to wrap expressions in the correct way ---

def as_tensor(expressions, indices = None):
    if indices is None:
        ufl_assert(isinstance(expressions, (list, tuple)),
            "Expecting nested list or tuple of Exprs.")
        return ListTensor(*expressions)
    if isinstance(indices, list):
        indices = tuple(indices)
    elif not isinstance(indices, tuple):
        indices = (indices,)
    if indices == ():
        return expressions
    return ComponentTensor(expressions, indices)

def as_matrix(expressions, indices = None):
    if indices is None:
        ufl_assert(isinstance(expressions, (list, tuple)),
            "Expecting nested list or tuple of Exprs.")
        ufl_assert(isinstance(expressions[0], (list, tuple)),
            "Expecting nested list or tuple of Exprs.")
        return ListTensor(*expressions)
    ufl_assert(len(indices) == 2, "Expecting exactly two indices.")
    return as_tensor(expressions, indices)

def as_vector(expressions, index = None):
    if index is not None:
        ufl_assert(isinstance(index, Index), "Expecting Index object.")
        index = (index,)
    return as_tensor(expressions, index)

def as_scalar(expression):
    ii = indices(expression.rank())
    if ii:
        expression = expression[ii]
    return expression, ii

def relabel(A, indexmap):
    "Relabel free indices of A with new indices, using the given mapping."
    ii = tuple(sorted(indexmap.keys()))
    jj = tuple(indexmap[i] for i in ii)
    ufl_assert(all(isinstance(i, Index) for i in ii), "Expecting Index objects.")
    ufl_assert(all(isinstance(j, Index) for j in jj), "Expecting Index objects.")
    return as_tensor(A, ii)[jj]

# --- Experimental support for dyadic notation:

def unit_list(i, n):
    return [(1 if i == j else 0) for j in xrange(n)]

def unit_list2(i, j, n):
    return [[(1 if (i == i0 and j == j0) else 0) for j0 in xrange(n)] for i0 in xrange(n)]

def unit_vector(i, d):
    return as_vector(unit_list(i, d))

def unit_vectors(d):
    return tuple(unit_vector(i, d) for i in range(d))

def unit_matrix(i, j, d):
    return as_matrix(unit_list2(i, j, d))

def unit_matrices(d):
    return tuple(unit_matrix(i, j, d) for i in range(d) for j in range(d))

def _test():
    #from ufl.tensors import unit_vector, unit_vectors, unit_matrix, unit_matrices
    from ufl.objects import triangle
    cell = triangle
    d = cell.d
    ei, ej, ek = unit_vectors(d)
    eii, eij, eik, eji, ejj, ejk, eki, ekj, ekk = unit_matrices(d)
    print ei
    print ej
    print ek
    print eii
    print eij
    print eik
    print eji
    print ejj
    print ejk
    print eki
    print ekj
    print ekk

