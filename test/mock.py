
from ufl.expr import Expr
from ufl import triangle

class Mock(Expr):
    def __init__(self, ops=(), sh=(), fi=(), ri=(), idim={}, c=triangle, r="", s="", h=0):
        Expr.__init__(self)
        self.ops = ops
        self.sh = sh
        self.fi = fi
        self.ri = ri
        self.idim = idim
        self.c = c
        self.r = r
        self.s = s
        self.h = h
        self.fields = ("ops", "sh", "fi", "ri", "idim", "c", "r", "s", "h")
    
    def operands(self):
        return self.ops
    
    def shape(self):
        return self.sh
    
    def rank(self):
        return len(self.shape())
    
    def cell(self):
        return self.c
    
    def free_indices(self):
        return self.fi
    
    def repeated_indices(self):
        return self.ri
    
    def index_dimensions(self):
        return self.idim
    
    def __repr__(self):
        return self.r
    
    def __str__(self):
        return self.s
    
    def __hash__(self):
        return self.h
    
    def get_fields(self):
        return tuple(getattr(self, f) for f in self.fields)
    
    def __eq__(self, other):
        for f in self.fields:
            a, b = getattr(self, f), getattr(other, f)
            if not a == b:
                print "self.%s != other.%s" % (f, f)
                print a, b
        return True

    def __nonzero__(self):
        return True 
    
    def __iter__(self):
        raise NotImplementedError

def _test():
    a = Mock(ops=(), sh=(1,), fi=(), ri=(), idim={}, c=triangle, r="", s="", h=0)
    b = Mock(ops=(), sh=(2,), fi=(), ri=(), idim={}, c=triangle, r="", s="", h=0)
    c = Mock(ops=(a, b), sh=(), fi=(), ri=(), idim={}, c=triangle, r="", s="", h=0)
    d = Mock(ops=(b, a), sh=(), fi=(), ri=(), idim={}, c=triangle, r="", s="", h=0)
    assert a == b
    assert c == d

if __name__ == "__main__":
    _test()

