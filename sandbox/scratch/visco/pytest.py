
class B(object):
    __slots__ = ()
    ccount = 0
    dcount = 0
    def __init__(self):
        B.ccount += 1
    def __del__(self):
        B.dcount += 1

import weakref

class A(B):
    #__slots__ = ('x','__weakref__') # This adds 8 bytes
    __slots__ = ('x',)

    #_cache = weakref.WeakValueDictionary()
    _cache = {}
    def __new__(cls, value):
        o = A._cache.get(value)
        if o is None:
            o = object.__new__(cls, value)
            A._cache[value] = o
        return o

    def __init__(self, value):
        if hasattr(self, 'x'):
            print 'has attr x'
        else:
            print 'has no attr x'
            B.__init__(self)
            self.x = value

    def __str__(self):
        return "A(%d)" % self.x

import sys

objects  = [A(v) for v in range(5)]
objects2 = [A(v) for v in range(5)]
print sys.getsizeof(objects[0])
print [id(o) for o in objects]
print [id(o) for o in objects2]
print [o for o in objects]
print [o for o in objects2]
print B.ccount, B.dcount
del objects
print B.ccount, B.dcount
del objects2
print B.ccount, B.dcount

"""
F:
18.9430084229
2.38723802567

J:
expand_indices took 57 s
557.466621399
57.3990740776

408.347160339
54.8738880157

expand_indices took 53 s

355.462684631
53.2244179249
"""

