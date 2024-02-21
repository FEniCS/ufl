"""Algorithm for expanding compound expressions."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs and Anders Logg
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2009-2010

import warnings


def apply_algebra_lowering(expr):
    """Expands high level compound operators to equivalent representations using basic operators."""
    warnings.warn(
        "The function apply_algebra_lowering is deprecated and will be removed after December 2024."
        " Please use expr.apply_algebra_lowering() directly instead.",
        FutureWarning,
    )
    return expr.apply_algebra_lowering()
