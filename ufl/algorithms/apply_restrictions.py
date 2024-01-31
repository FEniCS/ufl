"""Apply restrictions.

This module contains the apply_restrictions algorithm which propagates restrictions in a form
towards the terminals.
"""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import warnings


def apply_restrictions(expression):
    """Propagate restriction nodes to wrap differential terminals directly."""
    warnings.warn(
        "The function apply_default_restrictions is deprecated. "
        "Please, use object.apply_default_restrictions() directly instead.",
        FutureWarning)
    return expression.apply_restrictions()


def apply_default_restrictions(expression):
    """Some terminals can be restricted from either side.

    This applies a default restriction to such terminals if unrestricted.
    """
    warnings.warn(
        "The function apply_default_restrictions is deprecated. "
        "Please, use object.apply_default_restrictions() directly instead.",
        FutureWarning)
    return expression.apply_default_restrictions()
