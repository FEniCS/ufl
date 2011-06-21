
class Equation:

   def __init__(self, lhs, rhs):
       self.lhs = lhs
       self.rhs = rhs

   def __nonzero__(self):
       return repr(self.lhs) == repr(self.rhs)

   def __eq__(self, other):
       return isinstance(other, Equation) and\
           bool(self.lhs == other.lhs) and\
           bool(self.rhs == other.rhs)

   def __repr__(self):
       return "Equation(%r, %r)" % (self.lhs, self.rhs)

class Form:

   def __init__(self, a):
       self.a = a

   def __hash__(self):
       return hash(self.a)

   def __eq__(self, other):
       return Equation(self, other)

   def __repr__(self):
       return "Form(%r)" % (self.a,)

def test():
    forms = [Form(1), Form(2), Form(1), Form(0)]
    for i, f in enumerate(forms):
        for j, g in enumerate(forms):
            print '\nTesting forms', i, j, f, g
            print { f: 42, g: 84 }
            seq = f.a == g.a
            sne = f.a != g.a
            assert (f == g) == (f == g)
            assert bool((f == g) == (g == f)) == seq
            assert bool(f == g) == seq
            assert isinstance(eval(repr(f)), Form)
            assert isinstance(eval(repr(g)), Form)
            assert isinstance(eval(repr(f == g)), Equation)
            assert len({ f: 42, g: 84 }) == (1 if f == g else 2)

test()
