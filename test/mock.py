
from ufl.expr import Expr
from ufl import triangle

class Mock(Expr):
    def __init__(self, **kwargs):
        Expr.__init__(self)
        self.fields = []
        for k,v in kwargs.items():
            assert hasattr(Expr, k) # Invalid function name
            def foo(s):
                return getattr(s, "_%s" % k)
            setattr(self, "_%s" % k, v)
            setattr(self, k, foo)
            self.fields.append(k)
    
    def __eq__(self, other):
        fields = set(self.fields) | set(other.fields)
        for field in fields:
            a = getattr(self, field)(self)
            b = getattr(other, field)(other)
            if a != b:
                return False
        return True
    
    def __nonzero__(self):
        return True 
    
    def __iter__(self):
        raise NotImplementedError

def _test():
    a = Mock(operands=(), shape=(1,))
    b = Mock(operands=(), shape=(2,))
    c = Mock(operands=(a, b), shape=(), free_indices=(), index_dimensions={}, cell=triangle)
    d = Mock(operands=(b, a), shape=(), free_indices=(), index_dimensions={}, cell=triangle)
    assert a != b
    assert c == d

if __name__ == "__main__":
    _test()

