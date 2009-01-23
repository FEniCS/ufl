#!/usr/bin/env python
"""Run all tests"""

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2008-03-12 -- 2008-10-29"

from os import system
from glob import glob

# disable log output
import logging
logging.basicConfig(level=logging.CRITICAL)

# Tests to run
#tests = ["elements", "indices", "forms", "illegal", "algorithms"]
tests = [f.replace(".py", "") for f in glob("*.py")]
tests.remove("test")
tests.remove("run_pychecker")
tests.remove("analyse-demos")

# Run tests
for test in tests:
    print "Running tests: %s" % test
    system("python %s.py" % test)
    print ""

