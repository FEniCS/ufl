# coding: utf-8

import numbers

import ufl
from ufl.corealg.multifunction import MultiFunction
from ufl.corealg.map_dag import map_expr_dag
from ufl.core.multiindex import Index, FixedIndex
from ufl.form import Form
from ufl.algorithms import compute_form_data


class PrecedenceRules(MultiFunction):
    "An enum-like class for C operator precedence levels."

    def __init__(self):
        MultiFunction.__init__(self)

    def highest(self, o):
        return 0
    terminal = highest
    list_tensor = highest
    component_tensor = highest

    def restricted(self, o):
        return 5
    cell_avg = restricted
    facet_avg = restricted

    def call(self, o):
        return 10
    indexed = call
    min_value = call
    max_value = call
    math_function = call
    bessel_function = call

    def power(self, o):
        return 12

    def mathop(self, o):
        return 15
    derivative = mathop
    trace = mathop
    deviatoric = mathop
    cofactor = mathop
    skew = mathop
    sym = mathop

    def not_condition(self, o):
        return 20

    def product(self, o):
        return 30
    division = product
    # mod = product
    dot = product
    inner = product
    outer = product
    cross = product

    def add(self, o):
        return 40
    # sub = add
    index_sum = add

    def lt(self, o):
        return 50
    le = lt
    gt = lt
    ge = lt

    def eq(self, o):
        return 60
    ne = eq

    def and_condition(self, o):
        return 70

    def or_condition(self, o):
        return 71

    def conditional(self, o):
        return 72

    def lowest(self, o):
        return 80
    operator = lowest


_precrules = PrecedenceRules()


def precedence(expr):
    return _precrules(expr)


try:
    import colorama
    has_colorama = True
except ImportError:
    has_colorama = False


class UC:
    "An enum-like class for unicode characters."

    # Letters in this alphabet have contiguous code point numbers
    bold_math_a = u"𝐚"
    bold_math_A = u"𝐀"

    thin_space = u"\u2009"

    superscript_plus = u'⁺'
    superscript_minus = u'⁻'
    superscript_equals = u'⁼'
    superscript_left_paren = u'⁽'
    superscript_right_paren = u'⁾'
    superscript_digits = ["⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹"]

    subscript_plus = u'₊'
    subscript_minus = u'₋'
    subscript_equals = u'₌'
    subscript_left_paren = u'₍'
    subscript_right_paren = u'₎'
    subscript_digits = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"]

    sqrt = u'√'
    transpose = u'ᵀ'

    integral = u'∫'
    integral_double = u'∬'
    integral_triple = u'∭'
    integral_contour = u'∮'
    integral_surface = u'∯'
    integral_volume = u'∰'

    sum = u'∑'
    division_slash = '∕'
    partial = u'∂'
    epsilon = u'ε'
    omega = u'ω'
    Omega = u'Ω'
    gamma = u'γ'
    Gamma = u'Γ'
    nabla = u'∇'
    for_all = u'∀'

    dot = u'⋅'
    cross_product = u'⨯'
    circled_times = u'⊗'
    nary_product = u'∏'

    ne = u'≠'
    lt = u'<'
    le = u'≤'
    gt = u'>'
    ge = u'≥'

    logical_and = u'∧'
    logical_or = u'∨'
    logical_not = u'¬'

    element_of = u'∈'
    not_element_of = u'∉'

    left_white_square_bracket = u'⟦'
    right_white_squared_bracket = u'⟧'
    left_angled_bracket = u'⟨'
    right_angled_bracket = u'⟩'
    left_double_angled_bracket = u'⟪'
    right_double_angled_bracket = u'⟫'

    combining_right_arrow_above = '\u20D7'
    combining_overline = '\u0305'


def bolden_letter(c):
    if ord("A") <= ord(c) <= ord("Z"):
        c = chr(ord(c) - ord(u"A") + ord(UC.bold_math_A))
    elif ord("a") <= ord(c) <= ord("z"):
        c = chr(ord(c) - ord(u"a") + ord(UC.bold_math_a))
    return c


def superscript_digit(digit):
    return UC.superscript_digits[ord(digit) - ord("0")]


def subscript_digit(digit):
    return UC.subscript_digits[ord(digit) - ord("0")]


def bolden_string(s):
    return u"".join(bolden_letter(c) for c in s)


def overline_string(f):
    return u"".join("%s%s" % (c, UC.combining_overline) for c in f)


def subscript_number(number):
    assert isinstance(number, int)
    prefix = UC.subscript_minus if number < 0 else ''
    number = str(number)
    return prefix + ''.join(subscript_digit(c) for c in str(number))


def superscript_number(number):
    assert isinstance(number, int)
    prefix = UC.superscript_minus if number < 0 else ''
    number = str(number)
    return prefix + ''.join(superscript_digit(c) for c in str(number))


def opfont(opname):
    return bolden_string(opname)


def measure_font(dx):
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
    tdim = domain.topological_dimension()
    codim = integral_type_to_codim[integral_type]
    itgdim = tdim - codim

    # ipost = integral_postfixes[integral_type]
    istr = integral_by_dim[itgdim]

    # TODO: Render domain description

    if isinstance(subdomain_id, numbers.Integral):
        istr += subscript_number(int(subdomain_id))
    elif subdomain_id == "everywhere":
        pass
    elif subdomain_id == "otherwise":
        istr += "[rest of domain]"
    elif isinstance(subdomain_id, tuple):
        istr += ",".join([subscript_number(int(i)) for i in subdomain_id])

    dxstr = ufl.measure.integral_type_to_measure_name[integral_type]
    dxstr = measure_font(dxstr)

    return istr, dxstr


def par(s):
    return "(%s)" % s


def prec(expr):
    return 0  # FIXME
    # return precedence[expr._ufl_class_]


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def format_index(ii):
    if isinstance(ii, FixedIndex):
        s = "%d" % ii._value
    elif isinstance(ii, Index):
        s = "i%s" % subscript_number(ii._count)
    else:
        raise ValueError(f"Invalid index type {type(ii)}.")
    return s


def ufl2unicode(expression):
    "Generate Unicode string for a UFL expression or form."
    if isinstance(expression, Form):
        form_data = compute_form_data(expression)
        preprocessed_form = form_data.preprocessed_form
        return form2unicode(preprocessed_form, form_data)
    else:
        return expression2unicode(ufl.as_ufl(expression))


def expression2unicode(expression, argument_names=None, coefficient_names=None):
    rules = Expression2UnicodeHandler(argument_names, coefficient_names)
    return map_expr_dag(rules, expression)


def form2unicode(form, formdata):
    # formname = formdata.name
    argument_names = None
    coefficient_names = None

    # Define form as sum of integrals
    lines = []
    integrals = form.integrals()
    for itg in integrals:
        integrand_string = expression2unicode(
            itg.integrand(), argument_names, coefficient_names)

        istr, dxstr = get_integral_symbol(itg.integral_type(), itg.ufl_domain(), itg.subdomain_id())

        line = "%s %s %s" % (istr, integrand_string, dxstr)
        lines.append(line)

    return '\n  + '.join(lines)


def binop(expr, a, b, op, sep=" "):
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
    eprec = precedence(expr)
    aprec = precedence(expr.ufl_operands[0])
    op = opfont(opname)
    if aprec > eprec:
        arg = par(arg)
        sep = ""
    else:
        sep = UC.thin_space
    return "%s%s%s" % (op, sep, arg)


class Expression2UnicodeHandler(MultiFunction):
    def __init__(self, argument_names=None, coefficient_names=None, colorama_bold=False):
        MultiFunction.__init__(self)
        self.argument_names = argument_names
        self.coefficient_names = coefficient_names
        self.colorama_bold = colorama_bold and has_colorama

    # --- Terminal objects ---

    def scalar_value(self, o):
        if o.ufl_shape and self.colorama_bold:
            return "%s%s%s" % (colorama.Style.BRIGHT, o._value, colorama.Style.RESET_ALL)
        return "%s" % o._value

    def zero(self, o):
        if o.ufl_shape and self.colorama_bold:
            if len(o.ufl_shape) == 1:
                return "0%s" % UC.combining_right_arrow_above
            return "%s0%s" % (colorama.Style.BRIGHT, colorama.Style.RESET_ALL)
        return "0"

    def identity(self, o):
        if self.colorama_bold:
            return "%sI%s" % (colorama.Style.BRIGHT, colorama.Style.RESET_ALL)
        return "I"

    def permutation_symbol(self, o):
        if self.colorama_bold:
            return "%s%s%s" % (colorama.Style.BRIGHT, UC.epsilon, colorama.Style.RESET_ALL)
        return UC.epsilon

    def facet_normal(self, o):
        return "n%s" % UC.combining_right_arrow_above

    def spatial_coordinate(self, o):
        return "x%s" % UC.combining_right_arrow_above

    def argument(self, o):
        # Using ^ for argument numbering and _ for indexing since
        # indexing is more common than exponentiation
        if self.argument_names is None:
            i = o.number()
            bfn = "v" if i == 0 else "u"
            if not o.ufl_shape:
                return bfn
            elif len(o.ufl_shape) == 1:
                return "%s%s" % (bfn, UC.combining_right_arrow_above)
            elif self.colorama_bold:
                return "%s%s%s" % (colorama.Style.BRIGHT, bfn, colorama.Style.RESET_ALL)
            else:
                return bfn
        return self.argument_names[(o.number(), o.part())]

    def coefficient(self, o):
        # Using ^ for coefficient numbering and _ for indexing since
        # indexing is more common than exponentiation
        if self.coefficient_names is None:
            i = o.count()
            var = "w"
            if len(o.ufl_shape) == 1:
                var += UC.combining_right_arrow_above
            elif len(o.ufl_shape) > 1 and self.colorama_bold:
                var = "%s%s%s" % (colorama.Style.BRIGHT, var, colorama.Style.RESET_ALL)
            return "%s%s" % (var, subscript_number(i))
        return self.coefficient_names[o.count()]

    def constant(self, o):
        i = o.count()
        var = "c"
        if len(o.ufl_shape) == 1:
            var += UC.combining_right_arrow_above
        elif len(o.ufl_shape) > 1 and self.colorama_bold:
            var = "%s%s%s" % (colorama.Style.BRIGHT, var, colorama.Style.RESET_ALL)
        return "%s%s" % (var, superscript_number(i))

    def multi_index(self, o):
        return ",".join(format_index(i) for i in o)

    def label(self, o):
        return "l%s" % (subscript_number(o.count()),)

    # --- Non-terminal objects ---

    def variable(self, o, f, l):
        return "var(%s,%s)" % (f, l)

    def index_sum(self, o, f, i):
        if 1:  # prec(o.ufl_operands[0]) >? prec(o):
            f = par(f)
        return "%s[%s]%s" % (UC.sum, i, f)

    def sum(self, o, a, b):
        return binop(o, a, b, "+")

    def product(self, o, a, b):
        return binop(o, a, b, " ", sep="")

    def division(self, o, a, b):
        if is_int(b):
            b = subscript_number(int(b))
            if is_int(a):
                # Return as a fraction
                # NOTE: Maybe consider using fractional slash
                #  with normal numbers if terminals can handle it
                a = superscript_number(int(a))
            else:
                a = par(a)
            return "%s %s %s" % (a, UC.division_slash, b)
        return binop(o, a, b, UC.division_slash)

    def abs(self, o, a):
        return "|%s|" % (a,)

    def transposed(self, o, a):
        a = par(a)
        return "%s%s" % (a, UC.transpose)

    def indexed(self, o, A, ii):
        op0, op1 = o.ufl_operands
        Aprec = precedence(op0)
        oprec = precedence(o)
        if Aprec > oprec:
            A = par(A)
        return "%s[%s]" % (A, ii)

    def variable_derivative(self, o, f, v):
        f = par(f)
        v = par(v)
        nom = r"%s%s" % (UC.partial, f)
        denom = r"%s%s" % (UC.partial, v)
        return par(r"%s%s%s" % (nom, UC.division_slash, denom))

    def coefficient_derivative(self, o, f, w, v, cd):
        f = par(f)
        w = par(w)
        nom = r"%s%s" % (UC.partial, f)
        denom = r"%s%s" % (UC.partial, w)
        return par(r"%s%s%s[%s]" % (nom, UC.division_slash, denom, v))  # TODO: Fix this syntax...

    def grad(self, o, f):
        return mathop(o, f, "grad")

    def div(self, o, f):
        return mathop(o, f, "div")

    def nabla_grad(self, o, f):
        oprec = precedence(o)
        fprec = precedence(o.ufl_operands[0])
        if fprec > oprec:
            f = par(f)
        return "%s%s%s" % (UC.nabla, UC.thin_space, f)

    def nabla_div(self, o, f):
        oprec = precedence(o)
        fprec = precedence(o.ufl_operands[0])
        if fprec > oprec:
            f = par(f)
        return "%s%s%s%s%s" % (UC.nabla, UC.thin_space, UC.dot, UC.thin_space, f)

    def curl(self, o, f):
        oprec = precedence(o)
        fprec = precedence(o.ufl_operands[0])
        if fprec > oprec:
            f = par(f)
        return "%s%s%s%s%s" % (UC.nabla, UC.thin_space, UC.cross_product, UC.thin_space, f)

    def math_function(self, o, f):
        op = opfont(o._name)
        f = par(f)
        return "%s%s" % (op, f)

    def sqrt(self, o, f):
        f = par(f)
        return "%s%s" % (UC.sqrt, f)

    def exp(self, o, f):
        op = opfont("exp")
        f = par(f)
        return "%s%s" % (op, f)

    def atan2(self, o, f1, f2):
        f1 = par(f1)
        f2 = par(f2)
        op = opfont("arctan2")
        return "%s(%s, %s)" % (op, f1, f2)

    def bessel_j(self, o, nu, f):
        op = opfont("J")
        f = par(f)
        nu = subscript_number(int(nu))
        return "%s%s%s" % (op, nu, f)

    def bessel_y(self, o, nu, f):
        op = opfont("Y")
        f = par(f)
        nu = subscript_number(int(nu))
        return "%s%s%s" % (op, nu, f)

    def bessel_i(self, o, nu, f):
        op = opfont("I")
        f = par(f)
        nu = subscript_number(int(nu))
        return "%s%s%s" % (op, nu, f)

    def bessel_K(self, o, nu, f):
        op = opfont("K")
        f = par(f)
        nu = subscript_number(int(nu))
        return "%s%s%s" % (op, nu, f)

    def power(self, o, a, b):
        if is_int(b):
            b = superscript_number(int(b))
            return binop(o, a, b, "", sep="")
        return binop(o, a, b, "^", sep="")

    def outer(self, o, a, b):
        return binop(o, a, b, UC.circled_times)

    def inner(self, o, a, b):
        return "%s%s, %s%s" % (UC.left_angled_bracket, a, b, UC.right_angled_bracket)

    def dot(self, o, a, b):
        return binop(o, a, b, UC.dot)

    def cross(self, o, a, b):
        return binop(o, a, b, UC.cross_product)

    def determinant(self, o, A):
        return "|%s|" % (A,)

    def inverse(self, o, A):
        A = par(A)
        return "%s%s" % (A, superscript_number(-1))

    def trace(self, o, A):
        return mathop(o, A, "tr")

    def deviatoric(self, o, A):
        return mathop(o, A, "dev")

    def cofactor(self, o, A):
        return mathop(o, A, "cofac")

    def skew(self, o, A):
        return mathop(o, A, "skew")

    def sym(self, o, A):
        return mathop(o, A, "sym")

    def conj(self, o, a):
        # Overbar is already taken for average, and there is no superscript asterix in unicode.
        return mathop(o, a, "conj")

    def real(self, o, a):
        return mathop(o, a, "Re")

    def imag(self, o, a):
        return mathop(o, a, "Im")

    def list_tensor(self, o, *ops):
        l = ", ".join(ops)  # noqa: E741
        return "%s%s%s" % ("[", l, "]")

    def component_tensor(self, o, A, ii):
        return "[%s %s %s]" % (A, UC.for_all, ii)

    def positive_restricted(self, o, f):
        f = par(f)
        return "%s%s" % (f, UC.superscript_plus)

    def negative_restricted(self, o, f):
        f = par(f)
        return "%s%s" % (f, UC.superscript_minus)

    def cell_avg(self, o, f):
        f = overline_string(f)
        return f

    def facet_avg(self, o, f):
        f = overline_string(f)
        return f

    def eq(self, o, a, b):
        return binop(o, a, b, "=")

    def ne(self, o, a, b):
        return binop(o, a, b, UC.ne)

    def le(self, o, a, b):
        return binop(o, a, b, UC.le)

    def ge(self, o, a, b):
        return binop(o, a, b, UC.ge)

    def lt(self, o, a, b):
        return binop(o, a, b, UC.lt)

    def gt(self, o, a, b):
        return binop(o, a, b, UC.gt)

    def and_condition(self, o, a, b):
        return binop(o, a, b, UC.logical_and)

    def or_condition(self, o, a, b):
        return binop(o, a, b, UC.logical_or)

    def not_condition(self, o, a):
        a = par(a)
        return "%s%s" % (UC.logical_not, a)

    def conditional(self, o, c, t, f):
        c = par(c)
        t = par(t)
        f = par(t)
        If = opfont("if")
        Else = opfont("else")
        l = " ".join((t, If, c, Else, f))  # noqa: E741
        return l

    def min_value(self, o, a, b):
        op = opfont("min")
        return "%s(%s, %s)" % (op, a, b)

    def max_value(self, o, a, b):
        op = opfont("max")
        return "%s(%s, %s)" % (op, a, b)

    def expr_list(self, o, *ops):
        items = ", ".join(ops)
        return "%s %s %s" % (UC.left_white_square_bracket, items,
                             UC.right_white_squared_bracket)

    def expr_mapping(self, o, *ops):
        items = ", ".join(ops)
        return "%s %s %s" % (UC.left_double_angled_bracket, items,
                             UC.left_double_angled_bracket)

    def expr(self, o):
        raise ValueError("Missing handler for type %s" % str(type(o)))
