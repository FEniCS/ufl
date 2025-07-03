__authors__ = "Martin Sandve Alnæs"
__date__ = "2008-09-28 -- 2008-09-28"

import os
from glob import glob

import pytest

from ufl.algorithms import load_ufl_file, validate_form

demodir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demo"))


@pytest.mark.parametrize("filename", sorted(glob(os.path.join(demodir, "*.py"))))
def test_demo_files(filename):
    "Check each form in each file with validate_form."
    data = load_ufl_file(filename)
    for form in data.forms:
        validate_form(form)
