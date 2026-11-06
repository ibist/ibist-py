"""Structured results for confidence interval procedures."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConfidenceInterval:
    """Confidence limits for one named method, at full numerical precision."""

    method: str
    lower: float
    upper: float


@dataclass(frozen=True)
class CIResult:
    """An estimate and one or more confidence intervals for a parameter.

    ``conf_int`` has one row per method and columns lower, upper. ``to_frame``
    converts the intervals to a pandas DataFrame. Formatting a
    summary never modifies the stored values.
    """

    estimate: float
    intervals: tuple[ConfidenceInterval, ...]
    conf_level: float
    parameter: str
    null_value: float | None = None

    @property
    def methods(self) -> tuple[str, ...]:
        """Method labels in the order requested."""
        return tuple(interval.method for interval in self.intervals)

    @property
    def conf_int(self) -> np.ndarray:
        """A new array of lower and upper confidence limits."""
        return np.array([(ci.lower, ci.upper) for ci in self.intervals])

    def to_frame(self):
        """Return a pandas table with the estimate and method metadata."""
        frame = pd.DataFrame(self.conf_int, index=self.methods,
                             columns=["lower", "upper"])
        frame.index.name = "method"
        frame.attrs.update(estimate=self.estimate, conf_level=self.conf_level,
                           parameter=self.parameter, null_value=self.null_value)
        return frame

    def summary(self, digits: int = 3) -> str:
        """Format decimal places, using scientific notation for tiny values."""
        if (isinstance(digits, bool) or not isinstance(digits, int)
                or digits < 0):
            raise ValueError("digits must be a non-negative integer.")

        def number(value):
            if np.isfinite(value) and 0 < abs(value) < 0.5 * 10 ** -digits:
                return f"{value:.{digits}e}"
            return f"{value:.{digits}f}"

        lines = [f"{100 * self.conf_level:g}% confidence intervals for "
                 f"{self.parameter}", f"Estimate: {number(self.estimate)}", ""]
        width = max(len("method"), *(len(m) for m in self.methods))
        lines.append(f"{'method':<{width}}  {'lower':>12}  {'upper':>12}")
        lines.extend(f"{ci.method:<{width}}  {number(ci.lower):>12}  "
                     f"{number(ci.upper):>12}" for ci in self.intervals)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return self.summary()
