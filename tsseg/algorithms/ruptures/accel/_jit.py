"""numba is optional (``tsseg[accelerators]``): its decorators, or stand-ins."""

try:
    from numba import njit, prange

    AVAILABLE = True
except ImportError:
    AVAILABLE = False
    prange = range

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda func: func
