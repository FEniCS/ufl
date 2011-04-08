
def analyse_demo(name, shouldfail=False):
    import os
    from ufl.algorithms import load_ufl_file
    data = load_ufl_file(name)

    assert data, "Found no exported data in .ufl file."

    #for f in data.elements:
    #    validate(f)

    #for f in data.functions:
    #    validate(f)

    _msg_demofail = "The demo '%s' should fail validation, "\
                    "which means the form analysis is broken."
    from ufl.algorithms import validate_form
    for f in data.forms:
        failed = True
        try:
            validate_form(f)
            failed = False
        finally:
            if shouldfail:
                assert failed, _msg_demofail % os.path.basename(name)

def test_demos():
    import os, glob
    p = os.path.split(__file__)[0]
    
    files = glob.glob(os.path.join(p, "../../../demo/*.ufl"))
    assert files, "No main demos to test!"
    for name in files:
        yield analyse_demo, name
    
    files = glob.glob(os.path.join(p, "okdemos/*.ufl"))
    assert files, "No ok demos to test!"
    for name in files:
        yield analyse_demo, name
    
    files = glob.glob(os.path.join(p, "faildemos/*.ufl"))
    assert files, "No fail demos to test!"
    for name in files:
        yield analyse_demo, name, True

