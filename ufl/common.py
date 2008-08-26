"This module contains a collection of common utilities."

__authors__ = "Martin Sandve Alnes and Anders Logg"
__date__ = "2008-08-05 -- 2008-08-26"

import operator

def product(sequence):
    "Return the product of all elements in a sequence."
    return reduce(operator.__mul__, sequence, 1)

def some_key(a_dict):
    "Return an arbitrary key from a dictionary."
    return zip((0,), a_dict.iterkeys())[0][1]

class Counted(object):
    """A class of objects identified by a global counter.
    
    Requires that the subclass has a class variable
    _globalcount = 0"""
    __slots__ = ("_count",)
    def __init__(self, count = None):
        if count is None:
            self._count = self.__class__._globalcount
            self.__class__._globalcount += 1
        else:
            self._count = count
            if count >= self.__class__._globalcount:
                self.__class__._globalcount = count + 1


class Stack(list):
    "A stack datastructure."
    def __init__(self, *args):
        list.__init__(self, *args)
    
    def push(self, v):
        list.append(self, v)
    
    def peek(self):
        return self[-1]


class StackDict(dict):
    "A dict that can be changed incrementally with 'd.push(k,v)' and have changes rolled back with 'k,v = d.pop()'."
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self._l = []
    
    def push(self, k, v):
        self._l.append((k, self.get(k, None)))
        self[k] = v
    
    def pop(self):
        k, v = self._l.pop()
        if v is None:
            del self[k]
        else:
            self[k] = v
        return k, v


if __name__ == "__main__":
    d = StackDict(a=1)
    d.push("a", 2)
    d.push("a", 3)
    print d
    d.pop()
    print d
    d.pop()
    print d

