#!/usr/bin/env python

"""
This script simply runs the nose based tests,
rename to "test.py" when all tests are ported.
"""

if __name__ == "__main__":
    import os, subprocess
    os.chdir("newtests")
    subprocess.call("nosetests")

