# coding: utf-8

from __future__ import unicode_literals


import ufl
from ufl.corealg.multifunction import MultiFunction
from ufl.corealg.map_dag import map_expr_dag


from ufl.core.multiindex import Index, FixedIndex


integral = '\u222B'
integral_double = '\u222B'
integral_triple = '\u222B'
integral_contour = '\u222B'
integral_surface = '\u222B'
integral_volume = '\u222B'
sum = '\u2211'
division_slash = '\u2215'
partial = '\u2202'
epsilon = '\u03F5'
Omega = '\u03A9'
Gamma = '\u0393'
nabla = '\u2207'
for_all = '\u2200'

dot = '\u22C5'
cross_product = '\u2A2F'
circled_times = '\u2297'
ne = '\u2260'
lt = '<'
le = '\u2264'
gt = '>'
ge = '\u2265'

left_angled_bracket = '\u27E8'
right_angled_bracket = '\u27E9'

combining_right_arrow_above = '\u20D7'
combining_overline = '\u0305'

digit_superscript = ['\u2070', '\u00B9', '\u00B2', '\u00B3', '\u2074', '\u2075', '\u2076', '\u2077', '\u2078', '\u2079']
digit_subscript = ['\u2080', '\u2081', '\u2082', '\u2083', '\u2084', '\u2085', '\u2086', '\u2087', '\u2088', '\u2089']

postive_superscript = '\u207A'
negative_superscript = '\u207B'
postive_subscript = '\u208A'
negative_subscript = '\u208B'

sqrt = '\u221A'
transpose = '\u1D40'

try:
    import colorama
    has_colorama = True
except ImportError:
    has_colorama = False



def ufl2unicode(expression):
    "Generate Unicode string for a UFL expression or form."
    if isinstance(expression, Form):
        form_data = compute_form_data(expression)
        preprocessed_form = form_data.preprocessed_form
        return form2unicode(preprocessed_form, form_data)
    else:
        return expression2unicode(expression)


def expression2unicode(expression, argument_names=None, coefficient_names=None):
    rules = Expression2UnicodeHandler(argument_names, coefficient_names)
    return map_expr_dag(rules, expression)


def form2unicode(form, formdata):
    #formname = formdata.name
    argument_names = None
    coefficient_names = None

    # Define form as sum of integrals
    lines = []
    integrals = form.integrals()
    for itg in integrals:
        # TODO: Get list of expression strings instead of single
        # expression!
        integrand_string = expression2unicode(
            itg.integrand(),
            argument_names,
            coefficient_names)

        integral_type = itg.integral_type()
        istr = integral_symbols[integral_type]

        # domain = itg.ufl_domain()
        # TODO: Render domain description

        subdomain_id = itg.subdomain_id()
        if isinstance(subdomain_id, int):
            istr += subscript_number(subdomain_id)
        elif subdomain_id == "everywhere":
            pass
        elif subdomain_id == "otherwise":
            istr += "[oth]"
        elif isinstance(subdomain_id, tuple):
            istr += ",".join([subscript_number(i) for i in subdomain_id])

        dxstr = ufl.measure.integral_type_to_measure_name[integral_type]
        line = "%s %s  %s" % (istr, integrand_string, dxstr)
        lines.append(line)

    return '\n'.join(lines)

integral_symbols = {
    "cell": integral_volume,
    "exterior_facet": "%s[ext]" % integral_surface,
    "exterior_facet_bottom": "%s[ext, bottom]" % integral_surface,
    "exterior_facet_top": "%s[ext, top]" % integral_surface,
    "exterior_facet_vert": "%s[ext, vert]" % integral_surface,
    "interior_facet": "%s[int]" % integral_surface,
    "interior_facet_horiz": "%s[int, horiz]" % integral_surface,
    "interior_facet_vert": "%s[int, vert]" % integral_surface,
    "vertex": r"%s[vertex]" % integral,
    "custom": r"%s[custom]" % integral,
}


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
                return "0%s" % combining_right_arrow_above
            return "%s0%s" % (colorama.Style.BRIGHT, colorama.Style.RESET_ALL)
        return "0"

    def identity(self, o):
        if self.colorama_bold:
            return "%sI%s"  % (colorama.Style.BRIGHT, colorama.Style.RESET_ALL)
        return "I"

    def permutation_symbol(self, o):
        if self.colorama_bold:
             return "%s%s%s" % (colorama.Style.BRIGHT, epsilon, colorama.Style.RESET_ALL)
        return epsilon

    def facet_normal(self, o):
        return "n%s" % combining_right_arrow_above

    def argument(self, o):
        # Using ^ for argument numbering and _ for indexing since
        # indexing is more common than exponentiation
        if self.argument_names is None:
            bfn = bfname(o)
            if not o.ufl_shape:
                return bfn
            elif len(o.ufl_shape) == 1:
                return "%s%s" % (bfn, combining_right_arrow_above)
            elif self.colorama_bold:
                return "%s%s%s" % (colorama.Style.BRIGHT, bfn, colorama.Style.RESET_ALL)
            else:
                return bfn
        return self.argument_names[(o.number(), o.part())]

    def coefficient(self, o):
        # Using ^ for coefficient numbering and _ for indexing since
        # indexing is more common than exponentiation
        if self.coefficient_names is None:
            return cfname(o, self.colorama_bold)
        return self.coefficient_names[o.count()]
    constant = coefficient

    def multi_index(self, o):
        return format_multi_index(o)

    def variable(self, o):
        # TODO: Ensure variable has been handled
        e, l = o.ufl_operands
        return "s%s" % subscript_number(l._count)

    # --- Non-terminal objects ---

    def index_sum(self, o, f, i):
        return r"%s[%s]%s" % (sum, i, par(f))

    def sum(self, o, *ops):
        return " + ".join(par(op) for op in ops)

    def product(self, o, *ops):
        return " ".join(par(op) for op in ops)

    def division(self, o, a, b):
        if is_int(a) and is_int(b):
            # Return as a fraction
            # NOTE: Maybe consider using fractional slash
            #  with normal numbers if terminals can handle it
            return r"%s/%s" % (
                superscript_number(a),
                subscript_number(b))
        return r"%s%s%s" % (a, division_slash, b)

    def abs(self, o, a):
        return r"|%s|" % a

    def transposed(self, o, a):
        return "%s%s" % (par(a), transpose)

    def indexed(self, o, a, b):
        return "%s[%s]" % (par(a), b)

    def variable_derivative(self, o, f, v):
        nom = r"%s%s" % (partial, par(f))
        denom = r"%s%s" % (partial, par(v))
        return par(r"%s%s%s" % (nom, division_slash, denom))

    def coefficient_derivative(self, o, f, w, v):
        nom = r"%s%s" % (partial, par(f))
        denom = r"%s%s" % (partial, par(w))
        return par(r"%s%s%s[%s]" % (nom, division_slash, denom, v))  # TODO: Fix this syntax...

    def grad(self, o, f):
        return r"grad%s" % par(f)

    def div(self, o, f):
        return r"grad%s" % par(f)

    def nabla_grad(self, o, f):
        return r"%s%s" % (nabla, par(f))

    def nabla_div(self, o, f):
        return r"%s%s%s" % (nabla, dot, par(f))

    def curl(self, o, f):
        return r"%s%s%s" % (nabla, cross_product, par(f))

    def sqrt(self, o, f):
        return r"%s%s" % (sqrt, par(f))

    def exp(self, o, f):
        return "exp%s" % par(f)

    def ln(self, o, f):
        return r"ln%s" % par(f)

    def cos(self, o, f):
        return r"cos%s" % par(f)

    def sin(self, o, f):
        return r"sin%s" % par(f)

    def tan(self, o, f):
        return r"tan%s" % par(f)

    def cosh(self, o, f):
        return r"cosh%s" % par(f)

    def sinh(self, o, f):
        return r"sinh%s" % par(f)

    def tanh(self, o, f):
        return r"tanh%s" % par(f)

    def acos(self, o, f):
        return r"arccos%s" % par(f)

    def asin(self, o, f):
        return r"arcsin%s" % par(f)

    def atan(self, o, f):
        return r"arctan%s" % par(f)

    def atan2(self, o, f1, f2):
        return r"arctan2(%s, %s)" % (par(f1), par(f2))

    def erf(self, o, f):
        return r"erf%s" % par(f)

    def bessel_j(self, o, nu, f):
        return r"J[%s]%s" % (nu, par(f))

    def bessel_y(self, o, nu, f):
        return r"Y[%s]%s" % (nu, par(f))

    def bessel_i(self, o, nu, f):
        return r"I[%s]%s" % (nu, par(f))

    def bessel_K(self, o, nu, f):
        return r"K[%s]%s" % (nu, par(f))

    def power(self, o, a, b):
        if is_int(b):
            return "%s%s" % (par(a), superscript_number(b))
        return "%s^%s" % (par(a), par(b))

    def outer(self, o, a, b):
        return r"%s%s%s" % (par(a), circled_times, par(b))

    def inner(self, o, a, b):
        return "%s%s|%s%s" % (left_angled_bracket, a, b, right_angled_bracket)

    def dot(self, o, a, b):
        return r"%s%s%s" % (par(a), dot, par(b))

    def cross(self, o, a, b):
        return r"%s%s%s" % (par(a), cross_product, par(b))

    def trace(self, o, A):
        return "tr%s" % par(A)

    def determinant(self, o, A):
        return "|%s|" % A

    def inverse(self, o, A):
        return "%s%s" % (par(A), superscript_number(-1))

    def deviatoric(self, o, A):
        return "dev%s" % par(A)

    def cofactor(self, o, A):
        return "cofac%s" % par(A)

    def skew(self, o, A):
        return "skew%s" % par(A)

    def sym(self, o, A):
        return "sym%s" % par(A)

    def list_tensor(self, o):
        shape = o.ufl_shape
        if len(shape) == 1:
            ops = [self.visit(op) for op in o.ufl_operands]
            l = "\n  ".join(ops)
        elif len(shape) == 2:
            rows = []
            for row in o.ufl_operands:
                cols = (self.visit(op) for op in row.ufl_operands)
                rows.append(",  ".join(cols))
            l = "\n  ".join(rows)
        else:
            error("TODO: Unicode handler for list tensor of rank 3 or higher not implemented!")
        return "%s%s" % (l, transpose)

    def component_tensor(self, o, *ops):
        A, ii = ops
        return "[%s  %s %s]" % (A, for_all, ii)

    def positive_restricted(self, o, f):
        return "%s%s" % (par(f), postive_superscript)

    def negative_restricted(self, o, f):
        return "%s%s" % (par(f), negative_superscript)

    def cell_avg(self, o, f):
        # Put an overline over entire string
        ret = ""
        for ch in list(f):
            ret += "%s%s" % (ch, combining_overline)
        return ret

    def eq(self, o, a, b):
        return "(%s = %s)" % (a, b)

    def ne(self, o, a, b):
        return r"(%s %s %s)" % (a, ne, b)

    def le(self, o, a, b):
        return r"(%s %s %s)" % (a, le, b)

    def ge(self, o, a, b):
        return r"(%s %s %s)" % (a, ge, b)

    def lt(self, o, a, b):
        return r"(%s %s %s)" % (a, lt, b)

    def gt(self, o, a, b):
        return r"(%s %s %s)" % (a, gt, b)

    def and_condition(self, o, a, b):
        return "(%s && %s)" % (a, b)

    def or_condition(self, o, a, b):
        return "(%s || %s)" % (a, b)

    def not_condition(self, o, a):
        return "!(%s)" % (a,)

    def conditional(self, o, c, t, f):
        l = "{\n"
        l += "  %s, if %s\n" % (t, c)
        l += "  %s, otherwise\n" % f
        l += "}"
        return l

    def min_value(self, o, a, b):
        return "min(%s, %s)" % (a, b)

    def max_value(self, o, a, b):
        return "max(%s, %s)" % (a, b)

    def expr(self, o):
        error("Missing handler for type %s" % str(type(o)))


def digits_back_to_front(number):
    while number:
        digit = number % 10

        yield digit

        # remove last digit from number (as integer)
        number //= 10

def subscript_number(number):
    prefix = negative_subscript if number < 0 else ''
    return prefix + ''.join([
        digit_subscript[digit]
        for digit in reversed(list(digits_back_to_front(number)))
    ])

def superscript_number(number):
    prefix = negative_superscript if number < 0 else ''
    return prefix + ''.join([
        digit_superscript[digit]
        for digit in reversed(list(digits_back_to_front(number)))
    ])

def par(s, condition=True):  # TODO: Finish precedence handling by adding condition argument to calls to this function!
    if condition:
        return "(%s)" % s
    return str(s)

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
        error("Invalid index type %s." % type(ii))
    return s


def format_multi_index(ii, formatstring="%s"):
    return ",".join(formatstring % format_index(i) for i in ii)


def bfname(o):
    i = o.number()
    return "v" if i == 0 else "u"


def cfname(o, colorama_bold):
    i = o.count()
    var = "w"
    if len(o.ufl_shape) == 1:
        var += combining_right_arrow_above
    elif len(o.ufl_shape) > 1 and colorama_bold:
        var = "%s%s%s" % (colorama.Style.BRIGHT, var, colorama.Style.RESET_ALL)
    return "%s%s" % (var, superscript_number(i))
