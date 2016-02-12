#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from os.path import join as pjoin, split as psplit
import re
import sys
import platform

if sys.version_info < (2, 7):
    print("Python 2.7 or higher required, please upgrade.")
    sys.exit(1)

scripts = [pjoin("scripts", "ufl-analyse"),
           pjoin("scripts", "ufl-convert"),
           pjoin("scripts", "ufl-version"),
           pjoin("scripts", "ufl2py")]

if platform.system() == "Windows" or "bdist_wininst" in sys.argv:
    # In the Windows command prompt we can't execute Python scripts
    # without a .py extension. A solution is to create batch files
    # that runs the different scripts.
    batch_files = []
    for script in scripts:
        batch_file = script + ".bat"
        f = open(batch_file, "w")
        f.write('python "%%~dp0\%s" %%*' % psplit(script)[1])
        f.close()
        batch_files.append(batch_file)
    scripts.extend(batch_files)

version = re.findall('__version__ = "(.*)"',
                     open('ufl/__init__.py', 'r').read())[0]

setup(name="UFL",
      version = version,
      description = "Unified Form Language",
      author = "Martin Sandve Alnæs, Anders Logg",
      author_email = "fenics-dev@googlegroups.com",
      url = "http://www.fenicsproject.org",
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 2.7',
          'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Software Development :: Compilers',
          'Topic :: Software Development :: Libraries :: Python Modules',
          ],
      scripts = scripts,
      packages = [
          "ufl",
          "ufl.utils",
          "ufl.finiteelement",
          "ufl.core",
          "ufl.corealg",
          "ufl.algorithms",
          "ufl.formatting",
          ],
      package_dir = {"ufl": "ufl"},
      install_requires = ["numpy", "six"],
      data_files = [(pjoin("share", "man", "man1"),
                     [pjoin("doc", "man", "man1", "ufl-analyse.1.gz"),
                      pjoin("doc", "man", "man1", "ufl-convert.1.gz"),
                      pjoin("doc", "man", "man1", "ufl-version.1.gz"),
                      pjoin("doc", "man", "man1", "ufl2py.1.gz"),
                     ])])
