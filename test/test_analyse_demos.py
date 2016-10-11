# -*- coding: utf-8 -*-

from __future__ import print_function

__authors__ = "Martin Sandve Alnæs"
__date__ = "2008-09-28 -- 2008-09-28"

import os
import pytest
from ufl.algorithms import load_ufl_file, compute_form_data, validate_form
from glob import glob


demodir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demo"))


def get_demo_filenames():
    filenames = sorted(
        set(glob(os.path.join(demodir, "*.ufl")))
        - set(glob(os.path.join(demodir, "_*.ufl")))
        )
    return filenames


@pytest.mark.parametrize("filename", get_demo_filenames())
def test_demo_files(filename):
    "Check each form in each file with validate_form."
    data = load_ufl_file(filename)
    for form in data.forms:
        #fd = compute_form_data(form)  # TODO: Skip failure examples
        validate_form(form)
