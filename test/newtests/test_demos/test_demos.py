
def analyse_demo(demo):
    #from ufl import validate # TODO: Implement some generic function like validate that encompasses all validation that we can do in general for both elements, expressions, and forms.
    def validate(something): pass
    from ufl.algorithms import load_ufl_file
    data = load_ufl_file(demo)
    for d in data:
        validate(d)



        self.elements  = [] # alternative: {} # { name: FiniteElement }
        self.functions = [] # alternative: {} # { name: Function      }
        self.forms     = [] # alternative: {} # { name: Form          }

def test_demos():
    print "test_demos"
    import glob
    files = glob.glob("../../demo/*.ufl")
    assert files, "No demos to test!"
    for demo in files:
        yield analyse_demo, demo

