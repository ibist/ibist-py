"""Structured results for confidence interval procedures."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .asymcheck import AsymCheckResult


@dataclass(frozen=True)
class ConfidenceInterval:
    """Confidence limits for one named method, at full numerical precision."""

    method: str
    lower: float
    upper: float


@dataclass(frozen=True)
class CIResult:
    """An estimate and one or more confidence intervals for a parameter.

    ``conf_int`` has columns lower, upper. Scalar estimates have one row
    per method. Vector binomial estimates have rows ordered by input then
    method; ``inputs`` records each row's zero-based input index, x, and n.
    ``to_frame`` uses a (input, method) MultiIndex for vector inputs and the
    existing method index for scalar inputs. Formatting never modifies stored
    values. Two-sample differences have a scalar estimate and expose group
    counts through ``samples`` and ``sample_proportions``. Returned interval
    arrays, DataFrames, input tables, and sample proportions are copies.
    Optional diagnostics stores a separate AsymCheckResult.
    Supported CI functions retain original inputs and settings for
    asymcheck(result), even without attached diagnostics.
    """

    estimate: float | np.ndarray
    intervals: tuple[ConfidenceInterval, ...]
    conf_level: float
    parameter: str
    null_value: float | None = None
    alternative: str = "two.sided"
    correct: bool = False
    midp: bool = False
    _input_data: tuple[tuple[int, int, int], ...] = ()
    _sample_counts: tuple[tuple[int, int], ...] = ()
    _rate_data: tuple[tuple[int, float], ...] = ()
    diagnostics: "AsymCheckResult | None" = None
    _asymcheck_context: tuple = ()

    @property
    def side(self) -> str:
        """Interval form: two.sided, lower, or upper."""
        return {"two.sided": "two.sided", "greater": "lower",
                "less": "upper"}[self.alternative]

    @property
    def sample_rates(self) -> np.ndarray | None:
        """Fresh event rates for a Poisson-rate result."""
        if not self._rate_data:
            return None
        return np.array([x / exposure for x, exposure in self._rate_data])

    @property
    def sample_proportions(self) -> np.ndarray | None:
        """Fresh group proportions for an independent two-sample result."""
        if not self._sample_counts:
            return None
        return np.array([x / n for x, n in self._sample_counts])

    @property
    def samples(self) -> pd.DataFrame | None:
        """Fresh counts and binomial trial sizes or Poisson exposures."""
        if self._rate_data:
            frame = pd.DataFrame(self._rate_data, columns=["x", "exposure"])
            frame.index.name = "sample"
            return frame
        if not self._sample_counts:
            return None
        frame = pd.DataFrame(self._sample_counts, columns=["x", "n"])
        frame.index.name = "sample"
        return frame

    @property
    def inputs(self) -> pd.DataFrame | None:
        """Fresh per-interval input metadata, with zero-based input indices."""
        if not self._input_data:
            return None
        return pd.DataFrame(self._input_data, columns=["input", "x", "n"])

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
        vector = np.ndim(self.estimate) > 0
        if vector:
            index = pd.MultiIndex.from_arrays(
                [self.inputs["input"], self.methods], names=["input", "method"]
            )
        else:
            index = pd.Index(self.methods, name="method")
        frame = pd.DataFrame(self.conf_int, index=index, columns=["lower", "upper"])
        estimate = tuple(float(v) for v in self.estimate) if vector else self.estimate
        frame.attrs.update(estimate=estimate, conf_level=self.conf_level,
                           side=self.side,
                           parameter=self.parameter, null_value=self.null_value)
        if self._input_data:
            frame.attrs.update(alternative=self.alternative, correct=self.correct,
                               midp=self.midp, inputs=self._input_data)
        if self._sample_counts:
            frame.attrs.update(alternative=self.alternative, correct=self.correct,
                               samples=self._sample_counts)
        if self._rate_data:
            frame.attrs.update(alternative=self.alternative,
                               correct=self.correct, samples=self._rate_data)
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

        heading = (f"{100 * self.conf_level:g}% confidence intervals for "
                   f"{self.parameter}")
        headers = [heading]
        if self._input_data and np.ndim(self.estimate) == 0:
            _, count, size = self._input_data[0]
            headers.append(f"Data: x={count}, n={size}")
        if self._sample_counts:
            counts = tuple(x for x, _ in self._sample_counts)
            trials = tuple(n for _, n in self._sample_counts)
            headers.append(f"Data: x={counts}, n={trials}")
        if self._rate_data:
            counts = tuple(x for x, _ in self._rate_data)
            exposure = tuple(t for _, t in self._rate_data)
            headers.append(f"Data: x={counts}, exposure={exposure}")
        if self.alternative != "two.sided":
            bound = "lower" if self.alternative == "greater" else "upper"
            headers.append(f"Bound: {bound}")
        if np.ndim(self.estimate) > 0:
            frame = self.to_frame().reset_index()
            inputs = self.inputs
            frame.insert(2, "x", inputs["x"].to_numpy())
            frame.insert(3, "n", inputs["n"].to_numpy())
            frame.insert(4, "estimate", self.estimate[inputs["input"].to_numpy()])
            return "\n".join(headers) + "\n\n" + frame.to_string(
                index=False, float_format=number
            )
        lines = headers + [f"Estimate: {number(self.estimate)}", ""]
        width = max(len("method"), *(len(m) for m in self.methods))
        lines.append(f"{'method':<{width}}  {'lower':>12}  {'upper':>12}")
        lines.extend(f"{ci.method:<{width}}  {number(ci.lower):>12}  "
                     f"{number(ci.upper):>12}" for ci in self.intervals)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return self.summary()


def _ci_orientation(side=None, alternative=None):
    """Resolve the CI argument and its compatibility alias without guessing."""
    choices = ("two.sided", "lower", "upper")
    aliases = {"two.sided": "two.sided", "greater": "lower", "less": "upper"}
    if side is not None and (not isinstance(side, str) or side not in choices):
        raise ValueError("side must be two.sided, lower, or upper.")
    if alternative is not None:
        if not isinstance(alternative, str) or alternative not in aliases:
            raise ValueError("alternative must be two.sided, less, or greater.")
        legacy = aliases[alternative]
        if side is not None and side != legacy:
            raise ValueError(
                "side and alternative specify conflicting intervals.")
        side = legacy
    side = "two.sided" if side is None else side
    inverse = {value: key for key, value in aliases.items()}
    return side, inverse[side]
