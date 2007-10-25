

### maybe some support for ifs and switches?

class Conditional(UFLObject):
    def __init__(self):
        pass

class If(Conditional):
    def __init__(self, conditions, values):
        self.conditions = conditions
        self.values     = values
    
    def __str__(self):
        return "If(TODO)"

class Switch(Conditional):
    def __init__(self, argument, cases, values):
        self.argument = argument
        self.cases    = cases
        self.values   = values
    
    def __str__(self):
        return "Switch(TODO)"


