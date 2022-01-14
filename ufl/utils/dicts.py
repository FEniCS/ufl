# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Various dict manipulation utilities."""


def split_dict(d, criteria):
    """Split a dict d into two dicts based on a criteria on the keys."""
    a = {k: v for k, v in d.items() if criteria(k)}
    b = {k: v for k, v in d.items() if not criteria(k)}
    return a, b


def slice_dict(dictionary, keys, default=None):
    return tuple(dictionary.get(k, default) for k in keys)


def some_key(a_dict):
    """Return an arbitrary key from a dictionary."""
    return next(a_dict.keys())


def mergedicts(dicts):
    d = dict(dicts[0])
    for d2 in dicts[1:]:
        d.update(d2)
    return d


def mergedicts2(d1, d2):
    d = dict(d1)
    d.update(d2)
    return d


def subdict(superdict, keys):
    return dict((k, superdict[k]) for k in keys)


def dict_sum(items):
    """Construct a dict, in between dict(items) and sum(items), by accumulating items for each key."""
    d = {}
    for k, v in items:
        if k not in d:
            d[k] = v
        else:
            d[k] += v
    return d


class EmptyDictType(dict):
    def __setitem__(self, key, value):
        from ufl.log import error
        error("This is a frozen unique empty dictionary object, inserting values is an error.")

    def update(self, *args, **kwargs):
        from ufl.log import error
        error("This is a frozen unique empty dictionary object, inserting values is an error.")


EmptyDict = EmptyDictType()
