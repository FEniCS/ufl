"""Algorithms related to restrictions."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import warnings


def check_restrictions(expression, require_restriction):
    """Check that types that must be restricted are restricted in expression."""
    warnings.warn(
        "The function apply_default_restrictions is deprecated and will be removed after "
        "December 2024. Please use object.apply_default_restrictions() directly instead.",
        FutureWarning,
    )
    return expression.check_restrictions(require_restriction=require_restriction)
