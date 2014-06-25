

class A(object):
    __slots__ = ()

class B(object):
    __slots__ = ('y',)

class C(B, A):
    __slots__ = ('z',)

a = A()

b = B()
b.y = 3

c = C()
c.y = 1
c.z = 2

print(hasattr(A,'__slots__') and A.__slots__)
print(hasattr(B,'__slots__') and B.__slots__)
print(hasattr(C,'__slots__') and C.__slots__)
print(hasattr(A,'__dict__')  and A.__dict__)
print(hasattr(B,'__dict__')  and B.__dict__)
print(hasattr(C,'__dict__')  and C.__dict__)

print(hasattr(a,'__slots__') and a.__slots__)
print(hasattr(b,'__slots__') and b.__slots__)
print(hasattr(c,'__slots__') and c.__slots__)
print(hasattr(a,'__dict__')  and a.__dict__)
print(hasattr(b,'__dict__')  and b.__dict__)
print(hasattr(c,'__dict__')  and c.__dict__)

