"""Check arities."""
import warnings


class ArityMismatch(BaseException):
    """Arity mismatch exception."""

    pass


def _afmt(atuple):
    """Return a string representation of an arity tuple."""
    return tuple(f"conj({arg})" if conj else str(arg) for arg, conj in atuple)


def check_form_arity(form, arguments, complex_mode=False):
    """Check the arity of a form."""
    warnings.warn(
        "The function check_form_arity is deprecated and will be removed after December 2024. "
        "Please use form.check_arity() directly instead.",
        FutureWarning,
    )
    form.check_arity(arguments, complex_mode)
