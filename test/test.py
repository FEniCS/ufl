"""Run all tests"""

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2008-03-12 -- 2008-03-12"

from os import system

# disable log output
import logging
logging.basicConfig(level=logging.CRITICAL)


# Tests to run
tests = ["elements", "indices", "forms", "illegal", "algorithms"]

# Run tests
for test in tests:
    print "Running tests: %s" % test
    system("python %s.py" % test)
    print ""
