# -*- coding: utf-8 -*-
"Various utility data structures."

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


class Stack(list):
    """A stack datastructure."""

    def __init__(self, *args):
        list.__init__(self, *args)

    def push(self, v):
        list.append(self, v)

    def peek(self):
        return self[-1]


class StackDict(dict):
    """A dict that can be changed incrementally with 'd.push(k,v)' and have changes rolled back with 'k,v = d.pop()'."""

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self._l = []

    def push(self, k, v):
        """Store previous state for this key."""
        self._l.append((k, self.get(k, None)))
        if v is None:
            if k in self:
                del self[k]
        else:
            self[k] = v

    def pop(self):
        """Restore previous state for this key."""
        k, v = self._l.pop()
        if v is None:
            if k in self:
                del self[k]
        else:
            self[k] = v
        return k, v
