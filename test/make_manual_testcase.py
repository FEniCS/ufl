#!/usr/bin/env python
from __future__ import with_statement
import os
from os.path import join as pjoin
import re
from glob import glob

#--- Build list of code snippets found in tex files ---

path = "../doc/manual/chapters/"
filenames = glob(pjoin(path, "*.tex"))
codes = []
state = None
for fn in filenames:
    with open(fn) as f:
        for l in f:
            if state is None:
                if re.match(r"^\\begin\{code\}", l):
                    # TODO: Can get context information from this line by parsing comments?
                    if "TESTME" in l:
                        state = ""
            else:
                if re.match(r"^\\end\{code\}", l):
                    codes.append(state)
                    state = None
                else:
                    state += l

#--- Try executing all codes in an UFL environment ---

def execute_codes():
    code_prefix = """
    from ufl import *
    from ufl.classes import *

    from ufl.algorithms import *
    """

    code_suffix = """
    completed = True
    """

    failed = []
    for code in codes:
        print "-------------------------"
        print code

        namespace = {}
        try:
            exec (code_prefix + code + code_suffix) in namespace
        except:
            print "Code execution failed."

        completed = namespace.get("completed", False)
        if completed:
            print "SUCCESS"
        else:
            print "FAILED"
        failed.append(code)

#--- Generate unit test code ---

def indent(text, spaces=4):
    return "\n".join(spaces*" " + t for t in text.split("\n"))

testcode = """#!/usr/bin/env python

__authors__ = "Automatically generated from .tex files"
__date__ = "2009-02-07 -- 2009-02-07"

from ufltestcase import UflTestCase, main

from ufl import *
from ufl.classes import *
from ufl.algorithms import * 

class ManualTestCase(UflTestCase):

    def setUp(self):
        pass

%s

if __name__ == "__main__":
    main()
""" % "\n".join("    def test_%d(self):\n%s" % (i, indent(code, 8)) for (i, code) in enumerate(codes))

with open("manualtest.py", "w") as f:
    f.write(testcode)

