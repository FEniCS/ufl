#!/usr/bin/env python
"""Run all tests"""

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2008-03-12 -- 2009-02-18"

# Modified by Martin Alnes 2009

import unittest
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
#tests.remove("analyse-demos")
tests.remove("makemanualtestcase")

# Run tests
for test in tests:
    print "Running tests: %s" % test
    system("python %s.py" % test)
    print ""

# Run tests TODO: Make this work, to speed up test suite and get more compact output. Currently fails when instantiating testcases.
#all_tests = []
#for test in tests:
#    print "Adding tests from: %s" % test
#    module = __import__(test)
#    tests = [c() for c in vars(module).values() if isinstance(c, type) and issubclass(c, unittest.TestCase)]
#    all_tests.extend(tests)
#suite = unittest.TestSuite(all_tests)
#runner = unittest.TextTestRunner(suite)
#runner.run()

