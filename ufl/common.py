"This module contains a collection of common utilities."

__authors__ = "Martin Sandve Alnes and Anders Logg"
__date__ = "2008-08-05 -- 2008-08-18"

import operator

def product(sequence):
    return reduce(operator.__mul__, sequence, 1)

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

