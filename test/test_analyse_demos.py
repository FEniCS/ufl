# -*- coding: utf-8 -*-

__authors__ = "Martin Sandve Alnæs"
__date__ = "2008-09-28 -- 2008-09-28"

import os
import pytest
from ufl_legacy.algorithms import load_ufl_file, compute_form_data, validate_form
from glob import glob


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
