
def id_or_none(obj):
    """Returns None if the object is None, obj.ufl_id() if available, or id(obj) if not.

    This allows external libraries to implement an alternative
    to id(obj) in the ufl_id() function, such that ufl can identify
    objects as the same without knowing about their types.    
    """
    if obj is None:
        return None
    elif hasattr(obj, 'ufl_id'):
        return obj.ufl_id()
    else:
        #warning("Expecting an object implementing the ufl_id function.") # TODO: Can we enable this? Not sure about meshfunctions etc in dolfin.
        return id(obj)

def metadata_equal(a, b):
    return (sorted((k,id(v)) for k,v in a.items()) ==
            sorted((k,id(v)) for k,v in b.items()))

def metadata_hashdata(md):
    return tuple(sorted((k,id(v)) for k,v in md.items()))

