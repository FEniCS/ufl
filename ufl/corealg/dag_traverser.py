"""Base class for dag traversers."""

from abc import ABC, abstractmethod
from functools import singledispatchmethod, wraps

from ufl.classes import Expr


class DAGTraverser(ABC):
    """Base class for dag traversers."""

    def __init__(self, compress=True, vcache=None, rcache=None) -> None:
        """Initialise."""
        self._compress = compress
        # Temporary data structures
        # expr -> r = function(expr,...),  cache of intermediate results
        self._vcache = {} if vcache is None else vcache
        # r -> r,  cache of result objects for memory reuse
        self._rcache = {} if rcache is None else rcache

    def __call__(self, node: Expr, *args) -> Expr:
        """Perform memoised DAG traversal with ``process`` singledispatch method.

        Args:
            node: `Expr` to start DAG traversal from.
            args: arguments to the ``process`` singledispatchmethod.

        Returns:
            Processed `Expr`.

        """
        cache_key = (node, *args)
        try:
            return self._vcache[cache_key]
        except KeyError:
            result = self.process(node, *args)
            # Optionally check if r is in rcache, a memory optimization
            # to be able to keep representation of result compact
            if self._compress:
                result_ = self._rcache.get(result)
                if result_ is None:
                    # Cache miss: store in rcache
                    self._rcache[result] = result
                else:
                    # Cache hit: Use previously computed object result_,
                    # allowing result to be garbage collected as soon as possible
                    result = result_
            # Store result in cache
            self._vcache[cache_key] = result
            return result

    @singledispatchmethod
    @abstractmethod
    def process(self, node: Expr, *args) -> Expr:
        """Process node by type.

        Args:
            node: `Expr` to start DAG traversal from.
            args: arguments to the ``process`` singledispatchmethod.

        Returns:
            Processed `Expr`.

        """

    def reuse_if_untouched(self, node: Expr, *new_ufl_operands) -> Expr:
        """Reuse if touched.

        Args:
            node: `Expr` to start DAG traversal from.
            new_ufl_operands: new ufl_operands of ``node``.

        Returns:
            Processed `Expr`.

        """
        if all(nc == c for nc, c in zip(new_ufl_operands, node.ufl_operands)):
            return node
        else:
            return node._ufl_expr_reconstruct_(*new_ufl_operands)

    @staticmethod
    def postorder(method):
        """Suppress processed operands in arguments.

        It is more natural for users to write a post-order singledispatchmethod
        whose arguments are ``(self, o, *processed_operands, *additional_args)``,
        while `DAGTraverser` expects one whose arguments are
        ``(self, o, *additional_args)``.
        This decorator takes the former and converts the latter, processing
        ``o.ufl_operands`` behind the users.

        """
        @wraps(method)
        def wrapper(self, o, *args):
            processed_operands = [self(operand) for operand in o.ufl_operands]
            return method(self, o, *processed_operands, *args)
        return wrapper

    @staticmethod
    def postorder_only_children(indices):
        """Suppress processed operands corresponding to ``indices`` in arguments.

        This decorator is the same as `DAGTraverser.postorder` except that the
        decorated method only takes processed operands corresponding to ``indices``.

        """
        def postorder(method):
            @wraps(method)
            def wrapper(self, o, *args):
                processed_operands = [self(o.ufl_operands[i]) for i in indices]
                return method(self, o, *processed_operands, *args)
            return wrapper
        return postorder
