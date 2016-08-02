# -*- coding: utf-8 -*-
from __future__ import print_function

from setuptools import setup
from os.path import join as pjoin, split as psplit
import re
import sys
import platform
import codecs

if sys.version_info < (2, 7):
    print("Python 2.7 or higher required, please upgrade.")
    sys.exit(1)

# __init__.py has UTF-8 characters. Works in Python 2 and 3.
version = re.findall('__version__ = "(.*)"',
                     codecs.open('ufl/__init__.py', 'r',
                                 encoding='utf-8').read())[0]

url = "https://bitbucket.org/fenics-project/ufl/"
tarball = None
if 'dev' not in version:
    tarball = url + "downloads/ufl-%s.tar.gz" % version

script_names = ("ufl-analyse", "ufl-convert", "ufl-version", "ufl2py")
scripts = [pjoin("scripts", script) for script in script_names]
man_files = [pjoin("doc", "man", "man1", "%s.1.gz" % (script,)) for script in script_names]
data_files = [(pjoin("share", "man", "man1"), man_files)]

if platform.system() == "Windows" or "bdist_wininst" in sys.argv:
    # In the Windows command prompt we can't execute Python scripts
    # without a .py extension. A solution is to create batch files
    # that runs the different scripts.
    batch_files = []
    for script in scripts:
        batch_file = script + ".bat"
        with open(batch_file, "w") as f:
            f.write(sys.executable + ' "%%~dp0\%s" %%*' % psplit(script)[1])
        batch_files.append(batch_file)
    scripts.extend(batch_files)

setup(name="UFL",
      version=version,
      description="Unified Form Language",
      author="Martin Sandve Alnæs, Anders Logg",
      author_email="fenics-dev@googlegroups.com",
      url=url,
      download_url=tarball,
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
      scripts=scripts,
      packages=[
          "ufl",
          "ufl.utils",
          "ufl.finiteelement",
          "ufl.core",
          "ufl.corealg",
          "ufl.algorithms",
          "ufl.formatting",
      ],
      package_dir={"ufl": "ufl"},
      install_requires=["numpy", "six"],
      data_files=data_files
      )
