"""Teaching data and biostatistical inference for proportions and rates."""

__version__ = "0.1.0a1"

from . import datasets
from .agreement import agreement_test
from .association import AssociationTestResults, assoc_test
from .contingency import (NominalAssociationResult, TestResult, indep_test,
                          nominal_assoc)
from .common_odds import or_common_ci, or_common_test
from .ordinal import OrdinalAssociationResult, ordinal_assoc
from .paired import or_paired_ci, prop_paired_ci
from .homogeneity import or_homogeneity_test
from .odds_ratio import or_2x2_ci
from .proportions import prop_1s_ci, prop_2s_ci
from .rates import rate_1s_ci, rate_2s_ci, rate_test
from .risk_ratio import rr_2x2_ci
from .power import PowerResult, power_p1s_test, power_p2s_test
from .rate_power import power_r1s_test, power_r2s_test
from .trend import trend_test
from .results import CIResult, ConfidenceInterval

__all__ = ["CIResult", "ConfidenceInterval", "NominalAssociationResult",
           "PowerResult", "power_p1s_test", "power_p2s_test", "trend_test",
           "power_r1s_test", "power_r2s_test",
           "agreement_test",
           "AssociationTestResults", "assoc_test",
           "OrdinalAssociationResult", "ordinal_assoc",
           "TestResult", "datasets", "indep_test", "nominal_assoc",
           "or_2x2_ci", "or_common_ci", "or_common_test", "or_paired_ci",
           "prop_paired_ci", "or_homogeneity_test", "prop_1s_ci", "prop_2s_ci",
           "rate_1s_ci", "rate_2s_ci", "rate_test", "rr_2x2_ci"]
