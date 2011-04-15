#!/usr/bin/env python

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-09-28 -- 2008-09-28"

from ufltestcase import UflTestCase, main
import os

# Taken from http://ivory.idyll.org/blog/mar-07/replacing-commands-with-subprocess
from subprocess import Popen, PIPE, STDOUT
def get_status_output(cmd, input=None, cwd=None, env=None):
    pipe = Popen(cmd, shell=True, cwd=cwd, env=env, stdout=PIPE, stderr=STDOUT)

    (output, errout) = pipe.communicate(input=input)
    assert not errout

    status = pipe.returncode

    return (status, output)

from glob import glob

class DemoTestCase(UflTestCase):

    def setUp(self):
        super(DemoTestCase, self).setUp()
        #for f in glob("ufl_analyse_tmp_form*"):
        #    os.remove(f)
    
    def tearDown(self):
        #for f in glob("ufl_analyse_tmp_form*"):
        #    os.remove(f)
        super(DemoTestCase, self).tearDown()
    
    def _test_all_demos(self):
        # Check all at once
        skip = set(glob("../demo/_*.ufl"))
        filenames = [f for f in sorted(glob("../demo/*.ufl")) if not f in skip]
        cmd = "ufl-analyse %s" % " ".join(filenames)
        status, output = get_status_output(cmd)
        self.assertEqual(status, 0)

    def test_each_demo(self):
        status = 0

        skiplist = ()
        skiplist = glob("../demo/_*.ufl") #+ ["../demo/Hyperelasticity3D.ufl"]

        filenames = []
        for f in sorted(glob("../demo/*.ufl")):
            if f in skiplist:
                print "Skipping demo %s" % f
            else:
                filenames.append(f)

        # Check each file individually
        for f in filenames:
            cmd = "ufl-analyse %s" % f
            status, output = get_status_output(cmd)
            if status == 0:
                print "Successfully analysed %s without problems" % f
            else:
                name = "%s.analysis" % f
                print "Encountered problems when analysing %s "\
                      "(return code %s), see output in file %s" % (f, status, name)
                of = open(name, "w")
                of.write(output)
                of.close()
                print 
                print output
                print
        self.assertEqual(status, 0)

if __name__ == "__main__":
    main()
