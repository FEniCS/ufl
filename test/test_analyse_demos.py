#!/usr/bin/env python

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-09-28 -- 2008-09-28"

from ufltestcase import UflTestCase, main
import os
from ufl.algorithms import load_ufl_file, validate_form

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

    def get_demo_filenames(self):
        skiplist = glob("../demo/_*.ufl") #+ ["../demo/Hyperelasticity3D.ufl"]
        filenames = []
        for f in sorted(glob("../demo/*.ufl")):
            if f in skiplist:
                print "Skipping demo %s" % f
            else:
                filenames.append(f)
        return filenames

    def xtest_each_demo_with_ufl_analyse(self):
        "Check each file from cmdline with ufl-analyse."
        for f in self.get_demo_filenames():
            cmd = "ufl-analyse %s" % f
            status, output = get_status_output(cmd)
            self.assertEqual(status, 0)
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

    def test_each_demo_with_validate_form(self):
        "Check each form in each file with validate_form."
        for filename in self.get_demo_filenames():
            data = load_ufl_file(filename)
            for form in data.forms:
                try:
                    validate_form(form)
                    excepted = 0
                except:
                    excepted = 1
                self.assertEqual(excepted, 0)

if __name__ == "__main__":
    main()
