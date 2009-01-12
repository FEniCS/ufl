"""This module provides functions used by the UFL implementation to
output messages. These may be redirected by the user of UFL."""

__author__ = "Martin Sandve Alnaes and Anders Logg"
__date__ = "2005-02-04 -- 2009-01-12"
__copyright__ = "Copyright (C) 2005-2009 Anders Logg and Martin Sandve Alnaes"
__license__  = "GNU GPL version 3 or any later version"

import sys
import logging

__all__ = ["debug", "info", "warning", "error", "begin", "end",
           "set_level", "set_indent", "add_indent",
           "set_handler", "get_handler", "set_logger", "get_logger",
           "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Set up logger and handler
_log = logging.getLogger("ufl")
_handler = None

# Set initial indentation level
_indent_level = 0

# Import default log levels
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL

# Base class for UFL exceptions
class UFLException(Exception):
    "Base class for UFL exceptions"
    pass

#--- Functions for writing log messages ---

def debug(*message):
    "Write debug message."
    _log.debug(_format(*message))

def info(*message):
    "Write info message."
    text = _format_raw(*message)
    if len(text) >= 3 and text[-3:] == "...":
        # FIXME: Using print directly here, don't know how to write without newline using logging
        print _format(text),
        sys.stdout.flush()
    else:
        _log.info(_format(*message))

def warning(*message):
    "Write warning message."
    _log.warning(_format(*message))

def error(*message):
    "Write error message and raise an exception."
    _log.error(*message)
    raise UFLException(_format_raw(*message))

def begin(*message):
    "Begin task: write message and increase indentation level."
    info(*message)
    info("-"*len(_format_raw(*message)))
    add_indent()

def end():
    "End task: write a newline and decrease indentation level."
    info("")
    add_indent(-1)

def ufl_assert(condition, *message):
    "Assert that condition is true and otherwise issue an error with given message."
    if not condition:
        error(*message)

#--- Functions for controlling output ---

def set_level(level):
    "Set log level."
    _log.setLevel(level)

def set_indent(level):
    "Set indentation level."
    global _indent_level
    _indent_level = level

def add_indent(increment=1):
    "Add to indentation level."
    global _indent_level
    _indent_level += increment

#--- Functions for setting up logger and handler ---

def get_handler():
    "Get stream handler for logging."
    return _handler

def set_handler(handler):
    "Set stream handler for logging."
    global _handler
    if not _handler is None:
        _log.removeHandler(_handler)    
    _handler = handler
    _log.addHandler(_handler)

def get_logger():
    "Return message logger."
    return _log

def set_logger(logger):
    "Set message logger."
    global _logger
    _logger = logger

#--- Special functions used by logger ---

def _format(*message):
    "Format message including indentation."
    indent = 2*_indent_level*" "
    return indent + message[0] % message[1:]

def _format_raw(*message):
    "Format message without indentation."
    return message[0] % message[1:]
