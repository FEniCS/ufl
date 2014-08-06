
from ufl import *
from ufl.core.expr import Expr

class MockExpr(Expr):
    "A mock type for unit testing."
    def __init__(self, shape=None, free_indices=None, index_dimensions=None, cell=None):
        Expr.__init__(self)
        self.fields = []
        
        if not shape is None:
            self._shape = shape
            self.fields.append("shape")
        
        if not free_indices is None:
            self._free_indices = free_indices
            self.fields.append("free_indices")
        
        if not index_dimensions is None:
            self._index_dimensions = index_dimensions
            self.fields.append("index_dimensions")
        
        if not cell is None:
            self._cell = cell
            self.fields.append("cell")
    
    def shape(self):
        assert hasattr(self, "_shape")
        return self._shape
    
    def free_indices(self):
        assert hasattr(self, "_free_indices")
        return self._free_indices
        
    def index_dimensions(self):
        assert hasattr(self, "_index_dimensions")
        return self._index_dimensions
    
    def cell(self):
        assert hasattr(self, "_cell")
        return self._cell
    
    def matches(self, other):
        for field in self.fields:
            a = getattr(self, field)()
            b = getattr(other, field)()
            if not a == b:
                return False
        return True
    
    def __repr__(self):
        return "MockExpr(%s)" % ", ".join("%s=%s" % (k, repr(getattr(self, "_%s" % k))) for k in self.fields)
    
    def __iter__(self):
        raise NotImplementedError

def _test():
    a = MockExpr(shape=(1,))
    b = MockExpr(shape=(2,))
    assert not a.matches(b)
    
    i, j = indices(2)
    c = MockExpr(shape=(1, 2), free_indices=(i, j), index_dimensions={i:2, j:3}, cell=triangle)
    d = MockExpr(shape=(1, 2), free_indices=(i, j), index_dimensions={i:2, j:3}, cell=triangle)
    assert c.matches(d)
    
    e = FiniteElement("CG", triangle, 1)
    f = Coefficient(e)
    g = MockExpr(shape=(), free_indices=(), index_dimensions={}, cell=triangle)
    assert g.matches(f)
    h = MockExpr(shape=(1,), free_indices=(), index_dimensions={}, cell=triangle)
    assert not h.matches(f)

if __name__ == "__main__":
    _test()

