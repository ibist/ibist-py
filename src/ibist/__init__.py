"""Teaching data and biostatistical inference for proportions and rates."""

__version__ = "0.1.0a1"

from . import datasets
from .contingency import (NominalAssociationResult, TestResult, indep_test,
                          nominal_assoc)
from .homogeneity import or_homogeneity_test
from .odds_ratio import or_2x2_ci
from .proportions import prop_1s_ci
from .results import CIResult, ConfidenceInterval

__all__ = ["CIResult", "ConfidenceInterval", "NominalAssociationResult",
           "TestResult", "datasets", "indep_test", "nominal_assoc",
           "or_2x2_ci", "or_homogeneity_test", "prop_1s_ci"]
