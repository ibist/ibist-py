"""Chan--Zhang tail inversion with enclosing searches over all parameters.

Port of ibist-R/R/prop_2s_exact.R. Neither ordering nor tail acceptance is
assumed monotone. Numerical ambiguity is retained conservatively.
"""

import numpy as np
from scipy.stats import binom


def _independent_exact_limits(counts, sizes, tail, alternative, tol,
                               nuisance_tol):
    n1, n2 = (int(v) for v in sizes)
    if (n1 + 1) * (n2 + 1) > 1000000:
        raise ValueError(
            "Chan-Zhang requires at most 1,000,000 possible tables.")
    lower = (-1.0 if alternative == "less" else _lower(
        counts, sizes, tail, tol, nuisance_tol))
    upper = (1.0 if alternative == "greater" else -_lower(
        counts[::-1], sizes[::-1], tail, tol, nuisance_tol))
    return lower, upper


def _lower(counts, sizes, tail, tol, nuisance_tol):
    n1, n2 = (int(v) for v in sizes)
    estimate = counts[0] / n1 - counts[1] / n2
    if estimate == -1:
        return -1.0
    support = np.column_stack((np.tile(np.arange(n1 + 1), n2 + 1),
                               np.repeat(np.arange(n2 + 1), n1 + 1)))
    observed = int(counts[0]) + int(counts[1]) * (n1 + 1)
    differences = support[:, 0] / n1 - support[:, 1] / n2
    failures = sizes - support
    eps = np.finfo(float).eps

    def probabilities(delta):
        left = np.full(len(support), max(0.0, -delta))
        right = np.full(len(support), min(1.0, 1 - delta))
        for _ in range(52):
            q = (left + right) / 2
            p = np.column_stack((q + delta, q))
            p = np.clip(p, np.finfo(float).tiny, 1 - eps)
            with np.errstate(over="ignore"):
                slope = (support / p - failures / (1 - p)).sum(axis=1)
            positive = slope > 0
            left = np.where(positive, q, left)
            right = np.where(positive, right, q)
        return (np.column_stack((left + delta, left)),
                np.column_stack((right + delta, right)))

    def score_bounds(left, right, pl, pr):
        low = np.clip(np.column_stack((pl[0][:, 0], pr[0][:, 1])), 0, 1)
        high = np.clip(np.column_stack((pr[1][:, 0], pl[1][:, 1])), 0, 1)
        vlo = np.minimum(low * (1 - low), high * (1 - high))
        vhi = np.maximum(low * (1 - low), high * (1 - high))
        vhi[(low <= 0.5) & (high >= 0.5)] = 0.25
        vlo = np.sqrt((vlo / sizes).sum(axis=1))
        vhi = np.sqrt((vhi / sizes).sum(axis=1))

        def divide(numerator, denominator):
            with np.errstate(divide="ignore", invalid="ignore"):
                value = numerator / denominator
            return np.where((numerator == 0) & (denominator == 0), 0, value)

        numerator_low, numerator_high = differences - right, differences - left
        low_score = divide(numerator_low,
                           np.where(numerator_low >= 0, vhi, vlo))
        high_score = divide(numerator_high,
                            np.where(numerator_high >= 0, vlo, vhi))

        def corner(delta, sizes):
            if delta == 0:
                return 0.0
            with np.errstate(divide="ignore", invalid="ignore"):
                return (np.sqrt(sizes[1] * (-delta) / (1 + delta)) if delta < 0
                        else -np.sqrt(sizes[0] * delta / (1 - delta)))

        for index, nn in [(0, sizes), (-1, sizes[::-1])]:
            low_score[index] = corner(right, nn)
            high_score[index] = corner(left, nn)
        pad = 256 * eps * max(1, abs(low_score[observed]))
        if not np.isfinite(pad):
            pad = 0.0
        included = (high_score >= low_score[observed] - pad).reshape(
            n2 + 1, n1 + 1).T
        cutoff = np.max(np.where(included, np.arange(n2 + 1), -1), axis=1)
        return np.maximum.accumulate(cutoff)

    def possible(left, right, pl, pr):
        cutoff = score_bounds(left, right, pl, pr)
        low, high = max(0, -right), min(1, 1 - left)
        breaks = np.unique([low, high, max(low, min(high, 1 - right))])
        k = np.arange(n1 + 1)

        def probability(qs):
            values, slopes = [], []
            for start in range(0, len(qs), 256):
                q = qs[start:start + 256, None]
                p1 = np.clip(q + right, 0, 1)
                cdf = binom.cdf(cutoff, n2, q)
                mass = binom.pmf(k, n1, p1)
                value = (mass * cdf).sum(axis=1)
                slope = -n2 * (mass * binom.pmf(cutoff, n2 - 1, q)).sum(axis=1)
                derivative = n1 * (binom.pmf(k[:-1], n1 - 1, p1) *
                                   np.diff(cdf, axis=1)).sum(axis=1)
                slope += np.where(q[:, 0] + right < 1, derivative, 0)
                values.extend(value)
                slopes.extend(slope)
            return np.array(values), np.array(slopes)

        if np.any(probability(breaks)[0] > tail):
            return True
        l, r = breaks[:-1], breaks[1:]
        evaluations = 0
        while True:
            mid = (l + r) / 2
            values, slopes = probability(mid)
            if np.any(values > tail):
                return True
            h = (r - l) / 2
            total = n1 + n2
            upper = np.minimum(1, values + np.abs(slopes) * h +
                               total * (total - 1) * h**2 + 256 * eps)
            if np.all(upper <= tail):
                return False
            if upper.max() - values.max() <= nuisance_tol:
                return True
            keep = upper > tail
            l, r = np.concatenate((l[keep], mid[keep])), np.concatenate(
                (mid[keep], r[keep]))
            evaluations += len(mid)
            if evaluations > 1000000:
                raise ValueError("Nuisance maximization did not converge; "
                                 "increase nuisance_tol.")

    def search(left, right, pl, pr):
        if not possible(left, right, pl, pr):
            return None
        if right - left <= tol:
            return left
        middle = (left + right) / 2
        if middle == left or middle == right:
            return left
        pm = probabilities(middle)
        answer = search(left, middle, pl, pm)
        if answer is not None:
            return answer
        return search(middle, right, pm, pr)

    answer = search(-1.0, 1.0, probabilities(-1.0), probabilities(1.0))
    return 1.0 if answer is None else answer
