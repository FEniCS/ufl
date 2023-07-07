# -*- coding: utf-8 -*-
"Utilites for sorting."

# Copyright (C) 2008-2016 Johan Hake
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import warnings


def topological_sorting(nodes, edges):
    """
    Return a topologically sorted list of the nodes

    Implemented algorithm from Wikipedia :P

    <http://en.wikipedia.org/wiki/Topological_sorting>

    No error for cyclic edges...
    """

    L = []
    S = nodes[:]
    for node in nodes:
        for es in edges.values():
            if node in es and node in S:
                S.remove(node)
                continue

    while S:
        node = S.pop(0)
        L.append(node)
        node_edges = edges[node]
        while node_edges:
            m = node_edges.pop(0)
            found = False
            for es in edges.values():
                found = m in es
                if found:
                    break
            if not found:
                S.insert(0, m)

    return L


def sorted_by_count(seq):
    "Sort a sequence by the item.count()."
    return sorted(seq, key=lambda x: x.count())


def sorted_by_ufl_id(seq):
    "Sort a sequence by the item.ufl_id()."
    return sorted(seq, key=lambda x: x.ufl_id())


def sorted_by_key(mapping):
    "Sort dict items by key, allowing different key types."
    # Python3 doesn't allow comparing builtins of different type,
    # therefore the typename trick here
    def _key(x):
        return (type(x[0]).__name__, x[0])
    return sorted(mapping.items(), key=_key)


def canonicalize_metadata(metadata):
    """Assuming metadata to be a dict with string keys and builtin python types as values.

    Transform dict to a tuple of (key, value) item tuples ordered by key,
    with dict, list and tuple values converted the same way recursively.
    Lists and tuples are converted to tuples. Other values are converted using str().
    This is such that the end result can be hashed and sorted using regular <,
    because python 3 doesn't allow e.g. (3 < "auto") which occurs regularly in metadata.
    """
    if metadata is None:
        return ()

    if isinstance(metadata, dict):
        keys = sorted(metadata.keys())
        assert all(isinstance(key, str) for key in keys)
        values = [metadata[key] for key in keys]
    elif isinstance(metadata, (tuple, list)):
        values = metadata

    newvalues = []
    for value in values:
        if isinstance(value, (dict, list, tuple)):
            value = canonicalize_metadata(value)
        elif isinstance(value, (int, float, str)) or value is None:
            value = str(value)
        else:
            warnings.warn("Applying str() to a metadata value of type {0}, don't know if this is safe.".format(type(value).__name__))
            value = str(value)
        newvalues.append(value)

    if isinstance(metadata, dict):
        return tuple(zip(keys, newvalues))
    else:
        return tuple(newvalues)
