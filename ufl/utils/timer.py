# -*- coding: utf-8 -*-
"Timer utilites."

# Copyright (C) 2008-2015 Martin Sandve Alnæs and Anders Logg
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

from six.moves import xrange as range
import time

class Timer(object):
    def __init__(self, name):
        self.name = name
        self.times = []
        self('begin %s' % self.name)

    def __call__(self, msg):
        self.times.append((time.time(), msg))

    def end(self):
        self('end %s' % self.name)

    def __str__(self):
        line = "-"*60
        s = [line, "Timing of %s" % self.name]
        for i in range(len(self.times)-1):
            t = self.times[i+1][0] - self.times[i][0]
            msg = self.times[i][1]
            s.append("%9.2e s    %s" % (t, msg))
        s.append('Total time: %9.2e s' % (self.times[-1][0] - self.times[0][0]))
        s.append(line)
        return '\n'.join(s)
