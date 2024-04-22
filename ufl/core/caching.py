"""Caching.

Custom caching function to be used with class methods.
"""

__cache = {}


def cache(f):
    """Decorator for caching the result of a function."""

    def cached_f(self, *args, **kwargs):
        global __cache
        key = f"{args} {kwargs}"
        if self not in __cache[f.__name__]:
            __cache[f.__name__][self] = {}
        if key not in __cache[f.__name__][self]:
            __cache[f.__name__][self][key] = f(self, *args, **kwargs)
        return __cache[f.__name__][self][key]

    return cached_f


def initialise_cache(function_name: str):
    """Initialise a cache for a given function."""
    global __cache
    __cache[function_name] = {}


def clear_cache(function_name: str):
    """Remove the cache for a given function."""
    global __cache
    del __cache[function_name]
