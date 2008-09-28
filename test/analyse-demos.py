#!/usr/bin/env python

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-09-28 -- 2008-09-28"

import unittest

# disable log output
import logging
logging.basicConfig(level=logging.CRITICAL)

# Taken from http://ivory.idyll.org/blog/mar-07/replacing-commands-with-subprocess
from subprocess import Popen, PIPE, STDOUT
def get_status_output(cmd, input=None, cwd=None, env=None):
    pipe = Popen(cmd, shell=True, cwd=cwd, env=env, stdout=PIPE, stderr=STDOUT)

    (output, errout) = pipe.communicate(input=input)
    assert not errout

    status = pipe.returncode

    return (status, output)

from glob import glob

class DemoTestCase(unittest.TestCase):

    def setUp(self):
        pass
    
    def tearDown(self):
        for f in glob("ufl_analyse_tmp_form*"):
            os.remove(f)
    
    def test_something(self):
        result = 0
        for f in glob("../demo/*.ufl"):
            cmd = "ufl-analyse %s" % f
            status, output = get_status_output(cmd)
            if status == 0:
                print "Successfully analysed %s without problems" % f
            else:
                result = status
                name = "%s.analysis" % f
                print "Encountered problems when analysing %s (return code %s), see output in file %s" % (f, status, name)
                of = open(name, "w")
                of.write(output)
                of.close()
                print 
                print output
                print
        self.assertTrue(result == 0)

if __name__ == "__main__":
    unittest.main()
