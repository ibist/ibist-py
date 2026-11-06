"""Teaching data and biostatistical inference for proportions and rates."""

__version__ = "0.1.0a1"

from . import datasets
from .odds_ratio import or_2x2_ci
from .results import CIResult, ConfidenceInterval

__all__ = ["CIResult", "ConfidenceInterval", "datasets", "or_2x2_ci"]
