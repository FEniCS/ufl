"""This module provides functions used by the UFL implementation to
output messages. These may be redirected by the user of UFL."""

# Copyright (C) 2005-2013 Anders Logg and Martin Sandve Alnaes
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
#
# Modified by Johan Hake, 2009.
#
# First added:  2005-02-04
# Last changed: 2011-06-08

import sys
import types
import logging

log_functions = ["log", "debug", "info", "deprecate", "warning", "error", "begin", "end",
                 "set_level", "push_level", "pop_level", "set_indent", "add_indent",
                 "set_handler", "get_handler", "get_logger", "add_logfile", "set_prefix",
                 "info_red", "info_green", "info_blue",
                 "warning_red", "warning_green", "warning_blue"]

__all__ = log_functions + ["DEBUG", "INFO", "DEPRECATE", "WARNING", "ERROR",
                           "CRITICAL", "Logger", "log_functions"]

# Import default log levels
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL
DEPRECATE = (INFO + WARNING) // 2

# This is used to override emit() in StreamHandler for printing without newline
def emit(self, record):
    message = self.format(record)
    format_string = "%s" if getattr(record, "continued", False) else "%s\n"
    self.stream.write(format_string % message)
    self.flush()

# Colors
RED   = "\033[1;37;31m%s\033[0m"
BLUE  = "\033[1;37;34m%s\033[0m"
GREEN = "\033[1;37;32m%s\033[0m"

# Logger class
class Logger:

    def __init__(self, name, exception_type=Exception):
        "Create logger instance."
        self._name = name
        self._exception_type = exception_type

        # Set up handler
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(WARNING)
        # Override emit() in handler for indentation
        h.emit = types.MethodType(emit, h, h.__class__)
        self._handler = h

        # Set up logger
        self._log = logging.getLogger(name)
        assert len(self._log.handlers) == 0
        self._log.addHandler(h)
        self._log.setLevel(DEBUG)

        self._logfiles = {}

        # Set initial indentation level
        self._indent_level = 0

        # Setup stack with default logging level
        self._level_stack = [DEBUG]

        # Set prefix
        self._prefix = ""

    def add_logfile(self, filename=None, mode="a", level=DEBUG):
        if filename is None:
            filename = "%s.log" % self._name
        if filename in self._logfiles:
            self.warning("Trying to add logfile %s multiple times." % filename)
            return
        h = logging.FileHandler(filename, mode)
        h.emit = types.MethodType(emit, h, h.__class__)
        h.setLevel(level)
        self._log.addHandler(h)
        self._logfiles[filename] = h
        return h

    def get_logfile_handler(self, filename):
        return self._logfiles[filename]

    def log(self, level, *message):
        "Write a log message on given log level"
        text = self._format_raw(*message)
        if len(text) >= 3 and text[-3:] == "...":
            self._log.log(level, self._format(*message), extra={"continued": True})
        else:
            self._log.log(level, self._format(*message))

    def debug(self, *message):
        "Write debug message."
        self.log(DEBUG, *message)

    def info(self, *message):
        "Write info message."
        self.log(INFO, *message)

    def info_red(self, *message):
        "Write info message in red."
        self.log(INFO, RED % self._format_raw(*message))

    def info_green(self, *message):
        "Write info message in green."
        self.log(INFO, GREEN % self._format_raw(*message))

    def info_blue(self, *message):
        "Write info message in blue."
        self.log(INFO, BLUE % self._format_raw(*message))

    def deprecate(self, *message):
        "Write deprecation message."
        self.log(DEPRECATE, RED % self._format_raw(*message))

    def warning(self, *message):
        "Write warning message."
        self._log.warning(self._format(*message))

    def warning_red(self, *message):
        "Write warning message in red."
        self._log.warning(RED % self._format(*message))

    def warning_green(self, *message):
        "Write warning message in green."
        self._log.warning(GREEN % self._format(*message))

    def warning_blue(self, *message):
        "Write warning message in blue."
        self._log.warning(BLUE % self._format(*message))

    def error(self, *message):
        "Write error message and raise an exception."
        self._log.error(*message)
        raise self._exception_type(self._format_raw(*message))

    def begin(self, *message):
        "Begin task: write message and increase indentation level."
        self.info(*message)
        self.info("-"*len(self._format_raw(*message)))
        self.add_indent()

    def end(self):
        "End task: write a newline and decrease indentation level."
        self.info("")
        self.add_indent(-1)

    def push_level(self, level):
        "Push a log level on the level stack."
        self._level_stack.append(level)
        self.set_level(level)

    def pop_level(self):
        "Pop log level from the level stack, reverting to before the last push_level."
        self._level_stack.pop()
        level = self._level_stack[-1]
        self.set_level(level)

    def set_level(self, level):
        "Set log level."
        self._level_stack[-1] = level
        #self._log.setLevel(level)
        self._handler.setLevel(level)

    def set_indent(self, level):
        "Set indentation level."
        self._indent_level = level

    def add_indent(self, increment=1):
        "Add to indentation level."
        self._indent_level += increment

    def get_handler(self):
        "Get handler for logging."
        return self._handler

    def set_handler(self, handler):
        """Replace handler for logging.
        To add additional handlers instead
        of replacing the existing, use
        log.get_logger().addHandler(myhandler).
        See the logging module for more details.
        """
        self._log.removeHandler(self._handler)
        self._log.addHandler(handler)
        self._handler = handler
        handler.emit = types.MethodType(emit, self._handler, self._handler.__class__)

    def get_logger(self):
        "Return message logger."
        return self._log

    def set_prefix(self, prefix):
        "Set prefix for log messages."
        self._prefix = prefix

    def _format(self, *message):
        "Format message including indentation."
        indent = self._prefix + 2*self._indent_level*" "
        return "\n".join([indent + line for line in (message[0] % message[1:]).split("\n")])

    def _format_raw(self, *message):
        "Format message without indentation."
        return message[0] % message[1:]

#--- Set up global log functions ---

# Base class for UFL exceptions
class UFLException(Exception):
    "Base class for UFL exceptions"
    pass

ufl_logger = Logger("UFL", UFLException)

for foo in log_functions:
    exec("%s = ufl_logger.%s" % (foo, foo))
