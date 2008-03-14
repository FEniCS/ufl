#!/usr/bin/env python

"""
Functions used by the UFL implementation to provide output
messages that can be redirected by the user of the UFL library.
"""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-14-03"


class UFLException(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)


# TODO: Make redirection of all UFL messages easy. Use the python logging module, which seems meant for this kind of thing.
# (we may not need all four channels though)

def set_all_output(out):
    set_message_output(out)
    set_warning_output(out)
    set_error_output(out)
    set_debug_output(out)

def set_message_output(out):
    ufl_warning("set_message_output not implemented yet.")

def set_warning_output(out):
    ufl_warning("set_warning_output not implemented yet.")

def set_error_output(out):
    ufl_warning("set_error_output not implemented yet.")

def set_debug_output(out):
    ufl_warning("set_debug_output not implemented yet.")


def ufl_message(message):
    print message

def ufl_warning(message):
    print message

def ufl_error(message):
    print message
    raise UFLException(message)

def ufl_debug(message):
    print message


def ufl_assert(condition, message):
    if not condition:
        ufl_error(message)

