"This module contains a collection of common utilities."

__authors__ = "Martin Sandve Alnes and Anders Logg"
__date__ = "2008-08-05 -- 2008-08-14"

import operator
import operator

def product(sequence):
    return reduce(operator.__mul__, sequence, 1)

