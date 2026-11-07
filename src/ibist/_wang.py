"""Cached calls to the C++ Wang core shared with ibist-R."""

from functools import lru_cache

from ._wang_native import lower


@lru_cache(maxsize=128)
def _wang_lower(b, t, c, conf_level, precision, grid_one, grid_two):
    if (b + t + c + 1) * (b + t + c + 2) // 2 > 1000000:
        raise ValueError("Wang enumeration requires at most 1,000,000 states.")
    return lower(b, t, c, conf_level, precision, grid_one, grid_two)
