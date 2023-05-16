# -*- coding: utf-8 -*-

__authors__ = "Martin Sandve Alnæs"
__date__ = "2008-09-28 -- 2008-09-28"

import os
from glob import glob

import pytest

from ufl.algorithms import compute_form_data, load_ufl_file, validate_form

demodir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demo"))


def get_demo_filenames():
    filenames = sorted(
        set(glob(os.path.join(demodir, "*.py")))
        - set(glob(os.path.join(demodir, "_*.py")))
        )
    return filenames


@pytest.mark.parametrize("filename", get_demo_filenames())
def test_demo_files(filename):
    "Check each form in each file with validate_form."
    data = load_ufl_file(filename)
    for form in data.forms:
        validate_form(form)
