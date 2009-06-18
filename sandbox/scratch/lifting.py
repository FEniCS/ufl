
class LiftingResult(Operator):
    def __init__(self, operator, operand):
        Operator.__init__(self)
        self._operator = operator
        self._operand = operand

    def operands(self):
        return (self._operator, self._operand)
    # TODO: implement some functions from the Expr interface here

class LiftingOperatorResult(OperatorResult):
    def __init__(self, operator, operand):
        OperatorResult.__init__(self, operator, operand)

class LiftingFunctionResult(OperatorResult):
    def __init__(self, operator, operand):
        OperatorResult.__init__(self, operator, operand)

class TerminalOperator(Terminal):
    def __init__(self):
        Terminal.__init__(self)

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            a, = args
            if isinstance(a, Expr):
                return self._uflclass(self, a)
        return Terminal.__call__(self, *args, **kwargs)
    # TODO: implement some functions from the Expr interface here

class LiftingOperator(TerminalOperator):
    def __init__(self, element):
        TerminalOperator.__init__(self)
        self._element = element
    # TODO: implement some functions from the Expr interface here

class LiftingFunction(TerminalOperator):
    def __init__(self, element):
        TerminalOperator.__init__(self)
        self._element = element
    # TODO: implement some functions from the Expr interface here

cell = triangle
u_space = VectorElement("DG", cell, 1)
l_space = VectorElement("DG", cell, 0)
R = LiftingFunction(l_space)
r = LiftingOperator(l_space)

u = Function(u_space)

ru = r(u)
print ru

