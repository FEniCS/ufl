

class A(object):
    __slots__ = ('a',)
    def __init__(self):
        self.a = 1

class B(object):
    #__slots__ = ('b',)
    def __init__(self):
        self.b = 2

class C(A, B):
    __slots__ = ('c',)
    def __init__(self):
        self.c = 3

c = C()
print vars(c)

