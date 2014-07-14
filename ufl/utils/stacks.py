"Various utility data structures."

# Copyright (C) 2008-2014 Martin Sandve Alnes
#
# This file is part of UFL.
#
# UFL is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# UFL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UFL. If not, see <http://www.gnu.org/licenses/>.

class Stack(list):
    "A stack datastructure."
    def __init__(self, *args):
        list.__init__(self, *args)

    def push(self, v):
        list.append(self, v)

    def peek(self):
        return self[-1]

class StackDict(dict):
    "A dict that can be changed incrementally with 'd.push(k,v)' and have changes rolled back with 'k,v = d.pop()'."
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self._l = []

    def push(self, k, v):
        # Store previous state for this key
        self._l.append((k, self.get(k, None)))
        if v is None:
            if k in self:
                del self[k]
        else:
            self[k] = v

    def pop(self):
        # Restore previous state for this key
        k, v = self._l.pop()
        if v is None:
            if k in self:
                del self[k]
        else:
            self[k] = v
        return k, v
