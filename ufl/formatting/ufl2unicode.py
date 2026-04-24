"""UFL to unicode."""

import functools
import numbers
import warnings
from typing import Any

import ufl
from ufl.algorithms import compute_form_data
from ufl.core.multiindex import FixedIndex, Index
from ufl.corealg.dag_traverser import DAGTraverser
from ufl.form import Form

try:
    import colorama

    has_colorama = True
except ImportError:
    has_colorama = False

__all__ = _ = [
    "ufl2unicode",
    "PrecedenceRules",
    "precedence",
    "UC",
    "bolden_letter",
    "superscript_digit",
    "subscript_digit",
    "bolden_string",
    "overline_string",
    "subscript_number",
    "superscript_number",
    "opfont",
    "measure_font",
    "get_integral_symbol",
    "par",
    "is_int",
    "format_index",
    "Expression2UnicodeHandler",
]


class PrecedenceRules(DAGTraverser):
    """An enum-like class for C operator precedence levels."""

    @functools.singledispatchmethod
    def process(self, o: ufl.classes.Expr) -> int:
        """Process node by type."""
        return super().process(o)

    @process.register(ufl.classes.Terminal)
    @process.register(ufl.classes.ListTensor)
    @process.register(ufl.classes.ComponentTensor)
    def _(self, o: Any) -> int:
        # highest precendence
        return 0

    @process.register(ufl.classes.Restricted)
    @process.register(ufl.classes.CellAvg)
    @process.register(ufl.classes.FacetAvg)
    def _(self, o: Any) -> int:
        return 5

    @process.register(ufl.classes.Indexed)
    @process.register(ufl.classes.MinValue)
    @process.register(ufl.classes.MaxValue)
    @process.register(ufl.classes.MathFunction)
    @process.register(ufl.classes.BesselFunction)
    def _(self, o: Any) -> int:
        return 10

    @process.register(ufl.classes.Power)
    def _(self, o: Any) -> int:
        return 12

    @process.register(ufl.classes.Derivative)
    @process.register(ufl.classes.Trace)
    @process.register(ufl.classes.Deviatoric)
    @process.register(ufl.classes.Cofactor)
    @process.register(ufl.classes.Skew)
    @process.register(ufl.classes.Sym)
    def _(self, o: Any) -> int:
        return 15

    @process.register(ufl.classes.NotCondition)
    def _(self, o: ufl.classes.NotCondition) -> int:
        return 20

    @process.register(ufl.classes.Product)
    @process.register(ufl.classes.Division)
    @process.register(ufl.classes.Dot)
    @process.register(ufl.classes.Inner)
    @process.register(ufl.classes.Outer)
    @process.register(ufl.classes.Cross)
    def _(self, o: Any) -> int:
        return 30

    @process.register(ufl.classes.Sum)
    @process.register(ufl.classes.IndexSum)
    def _(self, o: Any) -> int:
        return 40

    @process.register(ufl.classes.LT)
    @process.register(ufl.classes.LE)
    @process.register(ufl.classes.GT)
    @process.register(ufl.classes.GE)
    def _(self, o: Any) -> int:
        return 50

    @process.register(ufl.classes.EQ)
    @process.register(ufl.classes.NE)
    def _(self, o: Any) -> int:
        return 60

    @process.register(ufl.classes.AndCondition)
    def _(self, o: ufl.classes.AndCondition) -> int:
        return 70

    @process.register(ufl.classes.OrCondition)
    def _(self, o: ufl.classes.OrCondition) -> int:
        return 71

    @process.register(ufl.classes.Conditional)
    def _(self, o: ufl.classes.Conditional) -> int:
        return 72

    @process.register(ufl.classes.Operator)
    def _(self, o: ufl.classes.Operator) -> int:
        return 80


_precrules = PrecedenceRules()


def precedence(expr):
    """Get the precedence of an expr."""
    return _precrules(expr)


class UC:
    """An enum-like class for unicode characters."""

    # Letters in this alphabet have contiguous code point numbers
    bold_math_a = "𝐚"
    bold_math_A = "𝐀"

    thin_space = "\u2009"

    superscript_plus = "⁺"
    superscript_minus = "⁻"
    superscript_equals = "⁼"
    superscript_left_paren = "⁽"
    superscript_right_paren = "⁾"
    superscript_digits = ["⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹"]

    subscript_plus = "₊"
    subscript_minus = "₋"
    subscript_equals = "₌"
    subscript_left_paren = "₍"
    subscript_right_paren = "₎"
    subscript_digits = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"]

    sqrt = "√"
    transpose = "ᵀ"

    integral = "∫"
    integral_double = "∬"
    integral_triple = "∭"
    integral_contour = "∮"
    integral_surface = "∯"
    integral_volume = "∰"

    sum = "∑"
    division_slash = "∕"
    partial = "∂"
    epsilon = "ε"
    omega = "ω"
    Omega = "Ω"
    gamma = "γ"
    Gamma = "Γ"
    nabla = "∇"
    for_all = "∀"

    dot = "⋅"
    cross_product = "⨯"
    circled_times = "⊗"
    nary_product = "∏"

    ne = "≠"
    lt = "<"
    le = "≤"
    gt = ">"
    ge = "≥"

    logical_and = "∧"
    logical_or = "∨"
    logical_not = "¬"

    element_of = "∈"
    not_element_of = "∉"

    left_white_square_bracket = "⟦"
    right_white_squared_bracket = "⟧"
    left_angled_bracket = "⟨"
    right_angled_bracket = "⟩"
    left_double_angled_bracket = "⟪"
    right_double_angled_bracket = "⟫"

    combining_right_arrow_above = "\u20d7"
    combining_overline = "\u0305"


def bolden_letter(c):
    """Bolden a letter."""
    if ord("A") <= ord(c) <= ord("Z"):
        c = chr(ord(c) - ord("A") + ord(UC.bold_math_A))
    elif ord("a") <= ord(c) <= ord("z"):
        c = chr(ord(c) - ord("a") + ord(UC.bold_math_a))
    return c


def superscript_digit(digit):
    """Make a digit superscript."""
    return UC.superscript_digits[ord(digit) - ord("0")]


def subscript_digit(digit):
    """Make a digit subscript."""
    return UC.subscript_digits[ord(digit) - ord("0")]


def bolden_string(s):
    """Bolden a string."""
    return "".join(bolden_letter(c) for c in s)


def overline_string(f):
    """Overline a string."""
    return "".join(f"{c}{UC.combining_overline}" for c in f)


def subscript_number(number):
    """Make a number subscript."""
    assert isinstance(number, int)
    prefix = UC.subscript_minus if number < 0 else ""
    number = str(number)
    return prefix + "".join(subscript_digit(c) for c in str(number))


def superscript_number(number):
    """Make a number superscript."""
    assert isinstance(number, int)
    prefix = UC.superscript_minus if number < 0 else ""
    number = str(number)
    return prefix + "".join(superscript_digit(c) for c in str(number))


def opfont(opname):
    """Use the font for operators."""
    return bolden_string(opname)


def measure_font(dx):
    """Use the font for measures."""
    return bolden_string(dx)


integral_by_dim = {3: UC.integral_triple, 2: UC.integral_double, 1: UC.integral, 0: UC.integral}

integral_type_to_codim = {
    "cell": 0,
    "exterior_facet": 1,
    "interior_facet": 1,
    "vertex": "tdim",
    "point": "tdim",
    "custom": 0,
    "overlap": 0,
    "interface": 1,
    "cutcell": 0,
}

integral_symbols = {
    "cell": UC.integral_volume,
    "exterior_facet": UC.integral_surface,
    "interior_facet": UC.integral_surface,
    "vertex": UC.integral,
    "point": UC.integral,
    "custom": UC.integral,
    "overlap": UC.integral,
    "interface": UC.integral,
    "cutcell": UC.integral,
}

integral_postfixes = {
    "cell": "",
    "exterior_facet": "ext",
    "interior_facet": "int",
    "vertex": "vertex",
    "point": "point",
    "custom": "custom",
    "overlap": "overlap",
    "interface": "interface",
    "cutcell": "cutcell",
}


def get_integral_symbol(integral_type, domain, subdomain_id):
    """Get the symbol for an integral."""
    tdim = domain.topological_dimension
    codim = integral_type_to_codim[integral_type]
    itgdim = tdim - codim

    # ipost = integral_postfixes[integral_type]
    istr = integral_by_dim[itgdim]

    # TODO: Render domain description

    subdomain_strs = []
    for subdomain in subdomain_id:
        if isinstance(subdomain, numbers.Integral):
            subdomain_strs.append(subscript_number(int(subdomain)))
        elif subdomain == "everywhere":
            pass
        elif subdomain == "otherwise":
            subdomain_strs.append("[rest of domain]")
    istr += ",".join(subdomain_strs)

    dxstr = ufl.measure.integral_type_to_measure_name[integral_type]
    dxstr = measure_font(dxstr)

    return istr, dxstr


def par(s):
    """Wrap in parentheses."""
    return f"({s})"


def is_int(s):
    """Check if a value is an integer."""
    try:
        int(s)
        return True
    except ValueError:
        return False


def format_index(ii):
    """Format an index."""
    if isinstance(ii, FixedIndex):
        s = f"{ii._value}"
    elif isinstance(ii, Index):
        s = "i{subscript_number(ii._count)}"
    else:
        raise ValueError(f"Invalid index type {type(ii)}.")
    return s


def ufl2unicode(expression):
    """Generate Unicode string for a UFL expression or form."""
    if isinstance(expression, Form):
        form_data = compute_form_data(expression)
        preprocessed_form = form_data.preprocessed_form
        return form2unicode(preprocessed_form, form_data)
    else:
        return expression2unicode(ufl.as_ufl(expression))


def expression2unicode(expression, argument_names=None, coefficient_names=None):
    """Generate Unicode string for a UFL expression."""
    rules = Expression2UnicodeHandler(argument_names, coefficient_names)
    return rules(expression)


def form2unicode(form, formdata):
    """Generate Unicode string for a UFL form."""
    argument_names = None
    coefficient_names = None

    # Define form as sum of integrals
    lines = []
    integrals = form.integrals()
    for itg in integrals:
        integrand_string = expression2unicode(itg.integrand(), argument_names, coefficient_names)

        istr, dxstr = get_integral_symbol(itg.integral_type(), itg.ufl_domain(), itg.subdomain_id())

        line = f"{istr} {integrand_string} {dxstr}"
        lines.append(line)

    return "\n  + ".join(lines)


def binop(expr, a, b, op, sep=" "):
    """Format a binary operation."""
    eprec = precedence(expr)
    op0, op1 = expr.ufl_operands
    aprec = precedence(op0)
    bprec = precedence(op1)
    # Assuming left-to-right evaluation, therefore >= and > here:
    if aprec >= eprec:
        a = par(a)
    if bprec > eprec:
        b = par(b)
    return sep.join((a, op, b))


def mathop(expr, arg, opname):
    """Format a math operation."""
    eprec = precedence(expr)
    aprec = precedence(expr.ufl_operands[0])
    op = opfont(opname)
    if aprec > eprec:
        arg = par(arg)
        sep = ""
    else:
        sep = UC.thin_space
    return f"{op}{sep}{arg}"


class Expression2UnicodeHandler(DAGTraverser):
    """Convert expressions to unicode."""

    def __init__(self, argument_names=None, coefficient_names=None, colorama_bold=False):
        """Initialise."""
        super().__init__()
        self.argument_names = argument_names
        self.coefficient_names = coefficient_names
        self.colorama_bold = colorama_bold and has_colorama

    @functools.singledispatchmethod
    def process(self, o: ufl.classes.Expr) -> str:
        """Process node by type."""
        warnings.warn(
            f"ufl2unicode does not define a handler for {type(o).__name__}, falling back to str()",
            stacklevel=2,
        )
        return str(o)

    # --- Terminal objects ---

    @process.register(ufl.classes.ScalarValue)
    def _(self, o: ufl.classes.ScalarValue) -> str:
        """Format a scalar_value."""
        if o.ufl_shape and self.colorama_bold:
            return f"{colorama.Style.BRIGHT}{o._value}{colorama.Style.RESET_ALL}"
        return f"{o._value}"

    @process.register(ufl.classes.Zero)
    def _(self, o: ufl.classes.Zero) -> str:
        """Format a zero."""
        if o.ufl_shape and self.colorama_bold:
            if len(o.ufl_shape) == 1:
                return f"0{UC.combining_right_arrow_above}"
            return f"{colorama.Style.BRIGHT}0{colorama.Style.RESET_ALL}"
        return "0"

    @process.register(ufl.classes.Identity)
    def _(self, o: ufl.classes.Identity) -> str:
        """Format a identity."""
        if self.colorama_bold:
            return f"{colorama.Style.BRIGHT}I{colorama.Style.RESET_ALL}"
        return "I"

    @process.register(ufl.classes.PermutationSymbol)
    def _(self, o: ufl.classes.PermutationSymbol) -> str:
        """Format a permutation_symbol."""
        if self.colorama_bold:
            return f"{colorama.Style.BRIGHT}{UC.epsilon}{colorama.Style.RESET_ALL}"
        return UC.epsilon

    @process.register(ufl.classes.FacetNormal)
    def _(self, o: ufl.classes.FacetNormal) -> str:
        """Format a facet_normal."""
        return f"n{UC.combining_right_arrow_above}"

    @process.register(ufl.classes.SpatialCoordinate)
    def _(self, o: ufl.classes.SpatialCoordinate) -> str:
        return f"x{UC.combining_right_arrow_above}"

    @process.register(ufl.classes.Argument)
    def _(self, o: ufl.classes.Argument) -> str:
        """Format an argument."""
        # Using ^ for argument numbering and _ for indexing since
        # indexing is more common than exponentiation
        if self.argument_names is None:
            i = o.number()
            bfn = "v" if i == 0 else "u"
            if not o.ufl_shape:
                return bfn
            elif len(o.ufl_shape) == 1:
                return f"{bfn}{UC.combining_right_arrow_above}"
            elif self.colorama_bold:
                return f"{colorama.Style.BRIGHT}{bfn}{colorama.Style.RESET_ALL}"
            else:
                return bfn
        return self.argument_names[(o.number(), o.part())]

    @process.register(ufl.classes.Coefficient)
    def _(self, o: ufl.classes.Coefficient) -> str:
        """Format a coefficient."""
        # Using ^ for coefficient numbering and _ for indexing since
        # indexing is more common than exponentiation
        if self.coefficient_names is None:
            i = o.count()
            var = "w"
            if len(o.ufl_shape) == 1:
                var += UC.combining_right_arrow_above
            elif len(o.ufl_shape) > 1 and self.colorama_bold:
                var = f"{colorama.Style.BRIGHT}{var}{colorama.Style.RESET_ALL}"
            return f"{var}{subscript_number(i)}"
        return self.coefficient_names[o.count()]

    @process.register(ufl.classes.Cofunction)
    def _(self, o: ufl.classes.Cofunction) -> str:
        """Format a cofunction."""
        if self.coefficient_names is None:
            i = o.count()
            var = "cofunction"
            if len(o.ufl_shape) == 1:
                var += UC.combining_right_arrow_above
            elif len(o.ufl_shape) > 1 and self.colorama_bold:
                var = f"{colorama.Style.BRIGHT}{var}{colorama.Style.RESET_ALL}"
            return f"{var}{subscript_number(i)}"
        return self.coefficient_names[o.count()]

    @process.register(ufl.classes.BaseFormOperator)
    def _(self, o: ufl.classes.BaseFormOperator) -> str:
        """Format a base_form_operator."""
        return "BaseFormOperator"

    @process.register(ufl.classes.Constant)
    def _(self, o: ufl.classes.Constant) -> str:
        """Format a constant."""
        i = o.count()
        var = "c"
        if len(o.ufl_shape) == 1:
            var += UC.combining_right_arrow_above
        elif len(o.ufl_shape) > 1 and self.colorama_bold:
            var = f"{colorama.Style.BRIGHT}{var}{colorama.Style.RESET_ALL}"
        return f"{var}{superscript_number(i)}"

    @process.register(ufl.classes.MultiIndex)
    def _(self, o: ufl.classes.MultiIndex) -> str:
        """Format a multi_index."""
        return ",".join(format_index(i) for i in o)

    @process.register(ufl.classes.Label)
    def _(self, o: ufl.classes.Label) -> str:
        """Format a label."""
        return f"l{subscript_number(o.count())}"

    # --- Non-terminal objects ---

    @process.register(ufl.classes.Action)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Action, a: str, b: str) -> str:
        """Format an Action."""
        return f"Action({a}, {b})"

    @process.register(ufl.classes.Variable)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Variable, f: str, a: str) -> str:
        """Format a variable."""
        return f"var({f},{a})"

    @process.register(ufl.classes.IndexSum)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.IndexSum, f: str, i: str) -> str:
        """Format a index_sum."""
        if 1:  # prec(o.ufl_operands[0]) >? prec(o):
            f = par(f)
        return f"{UC.sum}[{i}]{f}"

    @process.register(ufl.classes.Sum)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Sum, a: str, b: str) -> str:
        """Format a sum."""
        return binop(o, a, b, "+")

    @process.register(ufl.classes.Product)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Product, a: str, b: str) -> str:
        """Format a product."""
        return binop(o, a, b, " ", sep="")

    @process.register(ufl.classes.Division)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Division, a: str, b: str) -> str:
        """Format a division."""
        if is_int(b):
            b = subscript_number(int(b))
            if is_int(a):
                # Return as a fraction
                # NOTE: Maybe consider using fractional slash
                #  with normal numbers if terminals can handle it
                a = superscript_number(int(a))
            else:
                a = par(a)
            return f"{a} {UC.division_slash} {b}"
        return binop(o, a, b, UC.division_slash)

    @process.register(ufl.classes.Abs)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Abs, a: str) -> str:
        """Format an ans."""
        return f"|{a}|"

    @process.register(ufl.classes.Transposed)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Transposed, a: str) -> str:
        """Format a transposed."""
        a = par(a)
        return f"{a}{UC.transpose}"

    @process.register(ufl.classes.Indexed)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Indexed, A: str, ii: str) -> str:
        """Format an indexed."""
        op0, _op1 = o.ufl_operands
        Aprec = precedence(op0)
        oprec = precedence(o)
        if Aprec > oprec:
            A = par(A)
        return f"{A}[{ii}]"

    @process.register(ufl.classes.VariableDerivative)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.VariableDerivative, f: str, v: str) -> str:
        """Format a variable_derivative."""
        f = par(f)
        v = par(v)
        nom = f"{UC.partial}{f}"
        denom = f"{UC.partial}{v}"
        return par(f"{nom}{UC.division_slash}{denom}")

    @process.register(ufl.classes.CoefficientDerivative)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.CoefficientDerivative, f: str, w: str, v: str, cd: str) -> str:
        """Format a coefficient_derivative."""
        f = par(f)
        w = par(w)
        nom = f"{UC.partial}{f}"
        denom = f"{UC.partial}{w}"
        return par(f"{nom}{UC.division_slash}{denom}[{v}]")

    @process.register(ufl.classes.Grad)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Grad, f: str) -> str:
        """Format a grad."""
        return mathop(o, f, "grad")

    @process.register(ufl.classes.Div)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Div, f: str) -> str:
        """Format a div."""
        return mathop(o, f, "div")

    @process.register(ufl.classes.NablaGrad)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.NablaGrad, f: str) -> str:
        """Format a nabla_grad."""
        oprec = precedence(o)
        fprec = precedence(o.ufl_operands[0])
        if fprec > oprec:
            f = par(f)
        return f"{UC.nabla}{UC.thin_space}{f}"

    @process.register(ufl.classes.NablaDiv)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.NablaDiv, f: str) -> str:
        """Format a nabla_div."""
        oprec = precedence(o)
        fprec = precedence(o.ufl_operands[0])
        if fprec > oprec:
            f = par(f)
        return f"{UC.nabla}{UC.thin_space}{UC.dot}{UC.thin_space}{f}"

    @process.register(ufl.classes.Curl)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Curl, f: str) -> str:
        """Format a curl."""
        oprec = precedence(o)
        fprec = precedence(o.ufl_operands[0])
        if fprec > oprec:
            f = par(f)
        return f"{UC.nabla}{UC.thin_space}{UC.cross_product}{UC.thin_space}{f}"

    @process.register(ufl.classes.MathFunction)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.MathFunction, f: str) -> str:
        """Format a math_function."""
        op = opfont(o._name)
        return f"{op}{par(f)}"

    @process.register(ufl.classes.Sqrt)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Sqrt, f: str) -> str:
        """Format a sqrt."""
        return f"{UC.sqrt}{par(f)}"

    @process.register(ufl.classes.Exp)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Exp, f: str) -> str:
        """Format a exp."""
        op = opfont("exp")
        return f"{op}{par(f)}"

    @process.register(ufl.classes.Atan2)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Atan2, f1: str, f2: str) -> str:
        """Format a atan2."""
        f1 = par(f1)
        f2 = par(f2)
        op = opfont("arctan2")
        return f"{op}({f1}, {f2})"

    @process.register(ufl.classes.BesselJ)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.BesselJ, nu: str, f: str) -> str:
        """Format a bessel_j."""
        op = opfont("J")
        nu = subscript_number(int(nu))
        return f"{op}{nu}{par(f)}"

    @process.register(ufl.classes.BesselY)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.BesselY, nu: str, f: str) -> str:
        """Format a bessel_y."""
        op = opfont("Y")
        nu = subscript_number(int(nu))
        return f"{op}{nu}{par(f)}"

    @process.register(ufl.classes.BesselI)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.BesselI, nu: str, f: str) -> str:
        """Format a bessel_i."""
        op = opfont("I")
        nu = subscript_number(int(nu))
        return f"{op}{nu}{par(f)}"

    @process.register(ufl.classes.BesselK)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.BesselK, nu: str, f: str) -> str:
        """Format a bessel_K."""
        op = opfont("K")
        nu = subscript_number(int(nu))
        return f"{op}{nu}{par(f)}"

    @process.register(ufl.classes.Power)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Power, a: str, b: str) -> str:
        """Format a power."""
        if is_int(b):
            b = superscript_number(int(b))
            return binop(o, a, b, "", sep="")
        return binop(o, a, b, "^", sep="")

    @process.register(ufl.classes.Outer)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Outer, a: str, b: str) -> str:
        """Format an outer."""
        return binop(o, a, b, UC.circled_times)

    @process.register(ufl.classes.Inner)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Inner, a: str, b: str) -> str:
        """Format an inner."""
        return f"{UC.left_angled_bracket}{a}, {b}{UC.right_angled_bracket}"

    @process.register(ufl.classes.Dot)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Dot, a: str, b: str) -> str:
        """Format a dot."""
        return binop(o, a, b, UC.dot)

    @process.register(ufl.classes.Cross)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Cross, a: str, b: str) -> str:
        """Format a cross."""
        return binop(o, a, b, UC.cross_product)

    @process.register(ufl.classes.Determinant)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Determinant, A: str) -> str:
        """Format a determinant."""
        return f"|{A}|"

    @process.register(ufl.classes.Inverse)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Inverse, A: str) -> str:
        """Format an inverse."""
        A = par(A)
        return f"{A}{superscript_number(-1)}"

    @process.register(ufl.classes.Trace)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Trace, A: str) -> str:
        """Format a trace."""
        return mathop(o, A, "tr")

    @process.register(ufl.classes.Deviatoric)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Deviatoric, A: str) -> str:
        """Format a deviatoric."""
        return mathop(o, A, "dev")

    @process.register(ufl.classes.Cofactor)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Cofactor, A: str) -> str:
        """Format a cofactor."""
        return mathop(o, A, "cofac")

    @process.register(ufl.classes.Skew)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Skew, A: str) -> str:
        """Format a skew."""
        return mathop(o, A, "skew")

    @process.register(ufl.classes.Sym)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Sym, A: str) -> str:
        """Format a sym."""
        return mathop(o, A, "sym")

    @process.register(ufl.classes.Conj)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Conj, a: str) -> str:
        """Format a conj."""
        # Overbar is already taken for average, and there is no superscript asterix in unicode.
        return mathop(o, a, "conj")

    @process.register(ufl.classes.Real)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Real, a: str) -> str:
        """Format a real."""
        return mathop(o, a, "Re")

    @process.register(ufl.classes.Imag)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Imag, a: str) -> str:
        """Format a imag."""
        return mathop(o, a, "Im")

    @process.register(ufl.classes.ListTensor)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.ListTensor, *ops: str) -> str:
        """Format a list_tensor."""
        return f"[{', '.join(ops)}]"

    @process.register(ufl.classes.ComponentTensor)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.ComponentTensor, A: str, ii: str) -> str:
        """Format a component_tensor."""
        return f"[{A} {UC.for_all} {ii}]"

    @process.register(ufl.classes.PositiveRestricted)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.PositiveRestricted, f: str) -> str:
        """Format a positive_restriced."""
        return f"{par(f)}{UC.superscript_plus}"

    @process.register(ufl.classes.NegativeRestricted)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.NegativeRestricted, f: str) -> str:
        """Format a negative_restriced."""
        return f"{par(f)}{UC.superscript_minus}"

    @process.register(ufl.classes.CellAvg)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.CellAvg, f: str) -> str:
        """Format a cell_avg."""
        f = overline_string(f)
        return f

    @process.register(ufl.classes.FacetAvg)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.FacetAvg, f: str) -> str:
        """Format a facet_avg."""
        f = overline_string(f)
        return f

    @process.register(ufl.classes.EQ)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.EQ, a: str, b: str) -> str:
        """Format an eq."""
        return binop(o, a, b, "=")

    @process.register(ufl.classes.NE)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.NE, a: str, b: str) -> str:
        """Format a ne."""
        return binop(o, a, b, UC.ne)

    @process.register(ufl.classes.LE)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.LE, a: str, b: str) -> str:
        """Format a le."""
        return binop(o, a, b, UC.le)

    @process.register(ufl.classes.GE)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.GE, a: str, b: str) -> str:
        """Format a ge."""
        return binop(o, a, b, UC.ge)

    @process.register(ufl.classes.LT)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.LT, a: str, b: str) -> str:
        """Format a lt."""
        return binop(o, a, b, UC.lt)

    @process.register(ufl.classes.GT)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.GT, a: str, b: str) -> str:
        """Format a gt."""
        return binop(o, a, b, UC.gt)

    @process.register(ufl.classes.AndCondition)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.AndCondition, a: str, b: str) -> str:
        """Format an and_condition."""
        return binop(o, a, b, UC.logical_and)

    @process.register(ufl.classes.OrCondition)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.OrCondition, a: str, b: str) -> str:
        """Format an or_condition."""
        return binop(o, a, b, UC.logical_or)

    @process.register(ufl.classes.NotCondition)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.NotCondition, a: str, b: str) -> str:
        """Format a not_condition."""
        a = par(a)
        return f"{UC.logical_not}{a}"

    @process.register(ufl.classes.Conditional)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Conditional, c: str, t: str, f: str) -> str:
        """Format a conditional."""
        c = par(c)
        t = par(t)
        f = par(t)
        If = opfont("if")
        Else = opfont("else")
        return f"{t} {If} {c} {Else} {f}"

    @process.register(ufl.classes.MinValue)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.MinValue, a: str, b: str) -> str:
        """Format an min_value."""
        op = opfont("min")
        return f"{op}({a}, {b})"

    @process.register(ufl.classes.MaxValue)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.MaxValue, a: str, b: str) -> str:
        """Format an max_value."""
        op = opfont("max")
        return f"{op}({a}, {b})"

    @process.register(ufl.classes.ExprList)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.ExprList, *ops: str) -> str:
        """Format an expr_list."""
        items = ", ".join(ops)
        return f"{UC.left_white_square_bracket} {items} {UC.right_white_squared_bracket}"

    @process.register(ufl.classes.ExprMapping)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.ExprMapping, *ops: str) -> str:
        """Format an expr_mapping."""
        items = ", ".join(ops)
        return f"{UC.left_double_angled_bracket} {items} {UC.left_double_angled_bracket}"
