"Find potential bugs in UFL by static code analysis."

# PyChecker skips previously loaded modules
import os, sys, glob, shutil, re, logging, itertools
try:
    import numpy
except:
    pass
try:
    import swiginac
except:
    pass
try:
    import sympy
except:
    pass
try:
    import sympycore
except:
    pass

import pychecker.checker
import ufl

