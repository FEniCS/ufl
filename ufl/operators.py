"""This module extends the form language with free function operators,
which are either already available as member functions on UFL objects
or defined as compound operators involving basic operations on the UFL
objects."""

def Dx(o, i):
    """Return the partial derivative with respect to spatial variable number i"""
    return f.dx(i)
