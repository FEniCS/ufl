"""UFL to unicode."""

import numbers

import ufl
from ufl.algorithms import compute_form_data
from ufl.core.multiindex import FixedIndex, Index
from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction
from ufl.form import Form

try:
    import colorama
    has_colorama = True
except ImportError:
    has_colorama = False


class PrecedenceRules(MultiFunction):
    """An enum-like class for C operator precedence levels."""

    def __init__(self):
        """Initialise."""
        MultiFunction.__init__(self)

    def highest(self, o):
        """Return the highest precendence."""
        return 0
    terminal = highest
    list_tensor = highest
    component_tensor = highest

    def restricted(self, o):
        """Return precedence of a restriced."""
        return 5
    cell_avg = restricted
    facet_avg = restricted

    def call(self, o):
        """Return precedence of a call."""
        return 10
    indexed = call
    min_value = call
    max_value = call
    math_function = call
    bessel_function = call

    def power(self, o):
        """Return precedence of a power."""
        return 12

    def mathop(self, o):
        """Return precedence of a mathop."""
        return 15
    derivative = mathop
    trace = mathop
    deviatoric = mathop
    cofactor = mathop
    skew = mathop
    sym = mathop

    def not_condition(self, o):
        """Return precedence of a not_condition."""
        return 20

    def product(self, o):
        """Return precedence of a product."""
        return 30
    division = product
    # mod = product
    dot = product
    inner = product
    outer = product
    cross = product

    def add(self, o):
        """Return precedence of an add."""
        return 40
    # sub = add
    index_sum = add

    def lt(self, o):
        """Return precedence of a lt."""
        return 50
    le = lt
    gt = lt
    ge = lt

    def eq(self, o):
        """Return precedence of an eq."""
        return 60
    ne = eq

    def and_condition(self, o):
        """Return precedence of an and_condition."""
        return 70

    def or_condition(self, o):
        """Return precedence of an or_condition."""
        return 71

    def conditional(self, o):
        """Return precedence of a conditional."""
        return 72

    def lowest(self, o):
        """Return precedence of a lowest."""
        return 80
    operator = lowest


_precrules = PrecedenceRules()


def precedence(expr):
    """Get the precedence of an expr."""
    return _precrules(expr)


class UC:
    """An enum-like class for unicode characters."""

    # Letters in this alphabet have contiguous code point numbers
    bold_math_a = u"𝐚"
    bold_math_A = u"𝐀"

    thin_space = u"\u2009"

    superscript_plus = u"⁺"
    superscript_minus = u"⁻"
    superscript_equals = u"⁼"
    superscript_left_paren = u"⁽"
    superscript_right_paren = u"⁾"
    superscript_digits = ["⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹"]

    subscript_plus = u"₊"
    subscript_minus = u"₋"
    subscript_equals = u"₌"
    subscript_left_paren = u"₍"
    subscript_right_paren = u"₎"
    subscript_digits = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"]

    sqrt = u"√"
    transpose = u"ᵀ"

    integral = u"∫"
    integral_double = u"∬"
    integral_triple = u"∭"
    integral_contour = u"∮"
    integral_surface = u"∯"
    integral_volume = u"∰"

    sum = u"∑"
    division_slash = "∕"
    partial = u"∂"
    epsilon = u"ε"
    omega = u"ω"
    Omega = u"Ω"
    gamma = u"γ"
    Gamma = u"Γ"
    nabla = u"∇"
    for_all = u"∀"

    dot = u"⋅"
    cross_product = u"⨯"
    circled_times = u"⊗"
    nary_product = u"∏"

    ne = u"≠"
    lt = u"<"
    le = u"≤"
    gt = u">"
    ge = u"≥"

    logical_and = u"∧"
    logical_or = u"∨"
    logical_not = u"¬"

    element_of = u"∈"
    not_element_of = u"∉"

    left_white_square_bracket = u"⟦"
    right_white_squared_bracket = u"⟧"
    left_angled_bracket = u"⟨"
    right_angled_bracket = u"⟩"
    left_double_angled_bracket = u"⟪"
    right_double_angled_bracket = u"⟫"

    combining_right_arrow_above = "\u20D7"
    combining_overline = "\u0305"


def bolden_letter(c):
    """Bolden a letter."""
    if ord("A") <= ord(c) <= ord("Z"):
        c = chr(ord(c) - ord(u"A") + ord(UC.bold_math_A))
    elif ord("a") <= ord(c) <= ord("z"):
        c = chr(ord(c) - ord(u"a") + ord(UC.bold_math_a))
    return c


def superscript_digit(digit):
    """Make a digit superscript."""
    return UC.superscript_digits[ord(digit) - ord("0")]


def subscript_digit(digit):
    """Make a digit subscript."""
    return UC.subscript_digits[ord(digit) - ord("0")]


def bolden_string(s):
    """Bolden a string."""
    return u"".join(bolden_letter(c) for c in s)


def overline_string(f):
    """Overline a string."""
    return u"".join(f"{c}{UC.combining_overline}" for c in f)


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


integral_by_dim = {
    3: UC.integral_triple,
    2: UC.integral_double,
    1: UC.integral,
    0: UC.integral
}

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
    tdim = domain.topological_dimension()
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
        elif subdomain_id == "otherwise":
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
    return map_expr_dag(rules, expression)


def form2unicode(form, formdata):
    """Generate Unicode string for a UFL form."""
    argument_names = None
    coefficient_names = None

    # Define form as sum of integrals
    lines = []
    integrals = form.integrals()
    for itg in integrals:
        integrand_string = expression2unicode(
            itg.integrand(), argument_names, coefficient_names)

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


class Expression2UnicodeHandler(MultiFunction):
    """Convert expressions to unicode."""

    def __init__(self, argument_names=None, coefficient_names=None, colorama_bold=False):
        """Initialise."""
        MultiFunction.__init__(self)
        self.argument_names = argument_names
        self.coefficient_names = coefficient_names
        self.colorama_bold = colorama_bold and has_colorama

    # --- Terminal objects ---

    def scalar_value(self, o):
        """Format a scalar_value."""
        if o.ufl_shape and self.colorama_bold:
            return f"{colorama.Style.BRIGHT}{o._value}{colorama.Style.RESET_ALL}"
        return f"{o._value}"

    def zero(self, o):
        """Format a zero."""
        if o.ufl_shape and self.colorama_bold:
            if len(o.ufl_shape) == 1:
                return f"0{UC.combining_right_arrow_above}"
            return f"{colorama.Style.BRIGHT}0{colorama.Style.RESET_ALL}"
        return "0"

    def identity(self, o):
        """Format a identity."""
        if self.colorama_bold:
            return f"{colorama.Style.BRIGHT}I{colorama.Style.RESET_ALL}"
        return "I"

    def permutation_symbol(self, o):
        """Format a permutation_symbol."""
        if self.colorama_bold:
            return f"{colorama.Style.BRIGHT}{UC.epsilon}{colorama.Style.RESET_ALL}"
        return UC.epsilon

    def facet_normal(self, o):
        """Format a facet_normal."""
        return f"n{UC.combining_right_arrow_above}"

    def spatial_coordinate(self, o):
        """Format a spatial_coordinate."""
        return f"x{UC.combining_right_arrow_above}"

    def argument(self, o):
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

    def coefficient(self, o):
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

    def base_form_operator(self, o):
        """Format a base_form_operator."""
        return "BaseFormOperator"

    def constant(self, o):
        """Format a constant."""
        i = o.count()
        var = "c"
        if len(o.ufl_shape) == 1:
            var += UC.combining_right_arrow_above
        elif len(o.ufl_shape) > 1 and self.colorama_bold:
            var = f"{colorama.Style.BRIGHT}{var}{colorama.Style.RESET_ALL}"
        return f"{var}{superscript_number(i)}"

    def multi_index(self, o):
        """Format a multi_index."""
        return ",".join(format_index(i) for i in o)

    def label(self, o):
        """Format a label."""
        return f"l{subscript_number(o.count())}"

    # --- Non-terminal objects ---

    def variable(self, o, f, a):
        """Format a variable."""
        return f"var({f},{a})"

    def index_sum(self, o, f, i):
        """Format a index_sum."""
        if 1:  # prec(o.ufl_operands[0]) >? prec(o):
            f = par(f)
        return f"{UC.sum}[{i}]{f}"

    def sum(self, o, a, b):
        """Format a sum."""
        return binop(o, a, b, "+")

    def product(self, o, a, b):
        """Format a product."""
        return binop(o, a, b, " ", sep="")

    def division(self, o, a, b):
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

    def abs(self, o, a):
        """Format an ans."""
        return f"|{a}|"

    def transposed(self, o, a):
        """Format a transposed."""
        a = par(a)
        return f"{a}{UC.transpose}"

    def indexed(self, o, A, ii):
        """Format an indexed."""
        op0, op1 = o.ufl_operands
        Aprec = precedence(op0)
        oprec = precedence(o)
        if Aprec > oprec:
            A = par(A)
        return f"{A}[{ii}]"

    def variable_derivative(self, o, f, v):
        """Format a variable_derivative."""
        f = par(f)
        v = par(v)
        nom = f"{UC.partial}{f}"
        denom = f"{UC.partial}{v}"
        return par(f"{nom}{UC.division_slash}{denom}")

    def coefficient_derivative(self, o, f, w, v, cd):
        """Format a coefficient_derivative."""
        f = par(f)
        w = par(w)
        nom = f"{UC.partial}{f}"
        denom = f"{UC.partial}{w}"
        return par(f"{nom}{UC.division_slash}{denom}[{v}]")

    def grad(self, o, f):
        """Format a grad."""
        return mathop(o, f, "grad")

    def div(self, o, f):
        """Format a div."""
        return mathop(o, f, "div")

    def nabla_grad(self, o, f):
        """Format a nabla_grad."""
        oprec = precedence(o)
        fprec = precedence(o.ufl_operands[0])
        if fprec > oprec:
            f = par(f)
        return f"{UC.nabla}{UC.thin_space}{f}"

    def nabla_div(self, o, f):
        """Format a nabla_div."""
        oprec = precedence(o)
        fprec = precedence(o.ufl_operands[0])
        if fprec > oprec:
            f = par(f)
        return f"{UC.nabla}{UC.thin_space}{UC.dot}{UC.thin_space}{f}"

    def curl(self, o, f):
        """Format a curl."""
        oprec = precedence(o)
        fprec = precedence(o.ufl_operands[0])
        if fprec > oprec:
            f = par(f)
        return f"{UC.nabla}{UC.thin_space}{UC.cross_product}{UC.thin_space}{f}"

    def math_function(self, o, f):
        """Format a math_function."""
        op = opfont(o._name)
        return f"{op}{par(f)}"

    def sqrt(self, o, f):
        """Format a sqrt."""
        return f"{UC.sqrt}{par(f)}"

    def exp(self, o, f):
        """Format a exp."""
        op = opfont("exp")
        return f"{op}{par(f)}"

    def atan2(self, o, f1, f2):
        """Format a atan2."""
        f1 = par(f1)
        f2 = par(f2)
        op = opfont("arctan2")
        return f"{op}({f1}, {f2})"

    def bessel_j(self, o, nu, f):
        """Format a bessel_j."""
        op = opfont("J")
        nu = subscript_number(int(nu))
        return f"{op}{nu}{par(f)}"

    def bessel_y(self, o, nu, f):
        """Format a bessel_y."""
        op = opfont("Y")
        nu = subscript_number(int(nu))
        return f"{op}{nu}{par(f)}"

    def bessel_i(self, o, nu, f):
        """Format a bessel_i."""
        op = opfont("I")
        nu = subscript_number(int(nu))
        return f"{op}{nu}{par(f)}"

    def bessel_K(self, o, nu, f):
        """Format a bessel_K."""
        op = opfont("K")
        nu = subscript_number(int(nu))
        return f"{op}{nu}{par(f)}"

    def power(self, o, a, b):
        """Format a power."""
        if is_int(b):
            b = superscript_number(int(b))
            return binop(o, a, b, "", sep="")
        return binop(o, a, b, "^", sep="")

    def outer(self, o, a, b):
        """Format an outer."""
        return binop(o, a, b, UC.circled_times)

    def inner(self, o, a, b):
        """Format an inner."""
        return f"{UC.left_angled_bracket}{a}, {b}{UC.right_angled_bracket}"

    def dot(self, o, a, b):
        """Format a dot."""
        return binop(o, a, b, UC.dot)

    def cross(self, o, a, b):
        """Format a cross."""
        return binop(o, a, b, UC.cross_product)

    def determinant(self, o, A):
        """Format a determinant."""
        return f"|{A}|"

    def inverse(self, o, A):
        """Format an inverse."""
        A = par(A)
        return f"{A}{superscript_number(-1)}"

    def trace(self, o, A):
        """Format a trace."""
        return mathop(o, A, "tr")

    def deviatoric(self, o, A):
        """Format a deviatoric."""
        return mathop(o, A, "dev")

    def cofactor(self, o, A):
        """Format a cofactor."""
        return mathop(o, A, "cofac")

    def skew(self, o, A):
        """Format a skew."""
        return mathop(o, A, "skew")

    def sym(self, o, A):
        """Format a sym."""
        return mathop(o, A, "sym")

    def conj(self, o, a):
        """Format a conj."""
        # Overbar is already taken for average, and there is no superscript asterix in unicode.
        return mathop(o, a, "conj")

    def real(self, o, a):
        """Format a real."""
        return mathop(o, a, "Re")

    def imag(self, o, a):
        """Format a imag."""
        return mathop(o, a, "Im")

    def list_tensor(self, o, *ops):
        """Format a list_tensor."""
        return f"[{', '.join(ops)}]"

    def component_tensor(self, o, A, ii):
        """Format a component_tensor."""
        return f"[{A} {UC.for_all} {ii}]"

    def positive_restricted(self, o, f):
        """Format a positive_restriced."""
        return f"{par(f)}{UC.superscript_plus}"

    def negative_restricted(self, o, f):
        """Format a negative_restriced."""
        return f"{par(f)}{UC.superscript_minus}"

    def cell_avg(self, o, f):
        """Format a cell_avg."""
        f = overline_string(f)
        return f

    def facet_avg(self, o, f):
        """Format a facet_avg."""
        f = overline_string(f)
        return f

    def eq(self, o, a, b):
        """Format an eq."""
        return binop(o, a, b, "=")

    def ne(self, o, a, b):
        """Format a ne."""
        return binop(o, a, b, UC.ne)

    def le(self, o, a, b):
        """Format a le."""
        return binop(o, a, b, UC.le)

    def ge(self, o, a, b):
        """Format a ge."""
        return binop(o, a, b, UC.ge)

    def lt(self, o, a, b):
        """Format a lt."""
        return binop(o, a, b, UC.lt)

    def gt(self, o, a, b):
        """Format a gt."""
        return binop(o, a, b, UC.gt)

    def and_condition(self, o, a, b):
        """Format an and_condition."""
        return binop(o, a, b, UC.logical_and)

    def or_condition(self, o, a, b):
        """Format an or_condition."""
        return binop(o, a, b, UC.logical_or)

    def not_condition(self, o, a):
        """Format a not_condition."""
        a = par(a)
        return f"{UC.logical_not}{a}"

    def conditional(self, o, c, t, f):
        """Format a conditional."""
        c = par(c)
        t = par(t)
        f = par(t)
        If = opfont("if")
        Else = opfont("else")
        return " ".join((t, If, c, Else, f))

    def min_value(self, o, a, b):
        """Format an min_value."""
        op = opfont("min")
        return f"{op}({a}, {b})"

    def max_value(self, o, a, b):
        """Format an max_value."""
        op = opfont("max")
        return f"{op}({a}, {b})"

    def expr_list(self, o, *ops):
        """Format an expr_list."""
        items = ", ".join(ops)
        return f"{UC.left_white_square_bracket} {items} {UC.right_white_squared_bracket}"

    def expr_mapping(self, o, *ops):
        """Format an expr_mapping."""
        items = ", ".join(ops)
        return f"{UC.left_double_angled_bracket} {items} {UC.left_double_angled_bracket}"

    def expr(self, o):
        """Format an expr."""
        raise ValueError(f"Missing handler for type {type(o)}")
