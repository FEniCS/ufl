

class A(object):
    __slots__ = ()

class B(object):
    __slots__ = ('y',)

class C(B, A):
    __slots__ = ('z',)

c = C()
c.y = 1
c.z = 1
