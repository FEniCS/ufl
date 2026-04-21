"""Replace function spaces in all arguments."""

from functools import singledispatchmethod

from ufl import Argument
from ufl.algorithms.map_integrands import map_integrands
from ufl.classes import Expr
from ufl.corealg.dag_traverser import DAGTraverser


class FunctionSpaceReplacer(DAGTraverser):
    """Dispatcher."""

    def __init__(
        self,
        replacements: dict,
        part: int = 0,
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
    def _(self, o: Argument) -> Argument:
        """Apply to argument."""
        if o.ufl_function_space() in self.replacements:
            return Argument(self.replacements[o.ufl_function_space()], o._number, self.part)
        return self.reuse_if_untouched(o)


def replace_function_spaces(integrand, replacements: dict, offset):
    """Replace all instances of function spaces in an integrand.

    replacements should be a dictionary mapping from function spaces to
    what the spaces should be replaced with.
    """
    dag_traverser = FunctionSpaceReplacer(replacements, offset)
    return map_integrands(dag_traverser, integrand)
