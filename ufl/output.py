#!/usr/bin/env python

"""
Functions used by the UFL implementation to provide output
messages that can be redirected by the user of the UFL library.
"""

__authors__ = "Martin Sandve Alnes"
__date__    = "2008-03-14 -- 2008-05-15"

import logging
_log     = logging.getLogger("ufl")
_handler = logging.StreamHandler()
_log.addHandler(_handler)

class UFLException(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)

def get_handler():
    global _handler
    return _handler

def get_logger():
    return _log

def set_handler(handler):
    global _handler
    _log.removeHandler(_handler)
    _handler = handler
    _log.addHandler(_handler)

def ufl_debug(*message):
    _log.debug(*message)

def ufl_info(*message):
    _log.info(*message)

def ufl_warning(*message):
    _log.warning(*message)

def ufl_error(*message):
    _log.error(*message)
    text = message[0] % message[1:]
    raise UFLException(text)

def ufl_assert(condition, *message):
    if not condition:
        _log.error(*message)
        text = message[0] % message[1:]
        raise UFLException(text)
