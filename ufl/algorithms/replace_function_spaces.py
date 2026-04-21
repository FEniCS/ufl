"""Replace function spaces in all arguments."""

from functools import singledispatchmethod

from ufl import Argument
from ufl.algorithms.map_integrands import map_integrands
from ufl.classes import Expr
from ufl.corealg.dag_traverser import DAGTraverser
from ufl.functionspace import AbstractFunctionSpace


class FunctionSpaceReplacer(DAGTraverser):
    """Dispatcher."""

    def __init__(
        self,
        replacements: dict[AbstractFunctionSpace, AbstractFunctionSpace],
        part: int,
        compress: bool | None = True,
        visited_cache: dict[tuple, Expr] | None = None,
        result_cache: dict[Expr, Expr] | None = None,
    ) -> None:
        """Initialise."""
        super().__init__(compress=compress, visited_cache=visited_cache, result_cache=result_cache)
        self._dag_traverser_cache: dict[tuple[type, Expr, Expr, Expr], DAGTraverser] = {}
        self.replacements = replacements
        self.part = part

    @singledispatchmethod
    def process(self, o: Expr) -> Expr:
        """Process ``o``.

        Args:
            o: `Expr` to be processed.

        Returns:
            Processed object.

        """
        return super().process(o)

    @process.register(Expr)
    def _(self, o: Expr) -> Expr:
        """Do nothing."""
        return self.reuse_if_untouched(o)

    @process.register(Argument)
    def _(self, o: Argument) -> Expr:
        """Apply to argument."""
        if o.ufl_function_space() in self.replacements:
            return Argument(self.replacements[o.ufl_function_space()], o._number, self.part)
        return self.reuse_if_untouched(o)


def replace_function_spaces(
    integrand: Expr,
    replacements: dict[AbstractFunctionSpace, AbstractFunctionSpace],
    part: int = 0,
) -> Expr:
    """Replace all instances of function spaces in an integrand.

    Args:
        integrand: The integrand to do the replacements in.
        replacements: A dictionary mapping function spaces to
        the spaces they should be replaced with.
        part: The part to use in the replacement arguments.
    """
    dag_traverser = FunctionSpaceReplacer(replacements, part)
    return map_integrands(dag_traverser, integrand)
