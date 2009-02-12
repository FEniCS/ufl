"""This module provides functions used by the UFL implementation to
output messages. These may be redirected by the user of UFL."""

__author__ = "Martin Sandve Alnaes and Anders Logg"
__date__ = "2005-02-04 -- 2009-01-23"
__copyright__ = "Copyright (C) 2005-2009 Anders Logg and Martin Sandve Alnaes"
__license__  = "GNU GPL version 3 or any later version"

import sys
import logging

log_functions = ["debug", "info", "warning", "error", "begin", "end",
                 "set_level", "set_indent", "add_indent",
                 "set_handler", "get_handler", "get_logger"]

__all__ = log_functions + ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "Logger", "log_functions"]

# Import default log levels
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL

# Base class for UFL exceptions
class UFLException(Exception):
    "Base class for UFL exceptions"
    pass

# Logger class
class Logger:

    def __init__(self, name):
        "Create logger instance."

        # Set up logger and handler
        self._log = logging.getLogger(name)        
        self._handler = logging.StreamHandler()
        self._log.addHandler(self._handler)

        # Set initial indentation level
        self._indent_level = 0

    def debug(self, *message):
        "Write debug message."
        self._log.debug(self._format(*message))

    def info(self, *message):
        "Write info message."
        text = self._format_raw(*message)
        if len(text) >= 3 and text[-3:] == "...":
            # TODO: Using print directly here, don't know how to write without newline using logging
            print self._format(text),
            sys.stdout.flush()
        else:
            self._log.info(self._format(*message))

    def warning(self, *message):
        "Write warning message."
        self._log.warning(self._format(*message))

    def error(self, *message):
        "Write error message and raise an exception."
        self._log.error(*message)
        raise UFLException(self._format_raw(*message))

    def begin(self, *message):
        "Begin task: write message and increase indentation level."
        self.info(*message)
        self.info("-"*len(self._format_raw(*message)))
        self.add_indent()

    def end(self):
        "End task: write a newline and decrease indentation level."
        self.info("")
        self.add_indent(-1)

    def set_level(self, level):
        "Set log level."
        self._log.setLevel(level)
        
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

    def get_logger(self):
        "Return message logger."
        return self._log

    def _format(self, *message):
        "Format message including indentation."
        indent = 2*self._indent_level*" "
        return "\n".join([indent + line for line in (message[0] % message[1:]).split("\n")])

    def _format_raw(self, *message):
        "Format message without indentation."
        return message[0] % message[1:]

#--- Set up global log functions ---

ufl_logger = Logger("UFL")

for foo in log_functions:
    exec("%s = ufl_logger.%s" % (foo, foo))

