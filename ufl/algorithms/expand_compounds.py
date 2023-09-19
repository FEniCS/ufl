"""Algorithm for expanding compound expressions into equivalent representations using basic operators."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs and Anders Logg
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2009-2010

import warnings

from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering


def expand_compounds(e):
    """Expand compounds."""
    warnings.warn("The use of expand_compounds is deprecated and will be removed after December 2023. "
                  "Please, use apply_algebra_lowering directly instead", FutureWarning)

    return apply_algebra_lowering(e)
