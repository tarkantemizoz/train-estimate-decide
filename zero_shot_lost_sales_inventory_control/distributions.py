"""Compatibility for the legacy AER fitter's floating-point endpoint clamp."""

import math

from dynaplex.modelling import DiscreteDist


def fit_aer(mean: float, std: float) -> DiscreteDist:
    """Keep the native fitter except where legacy Mix clamps an endpoint.

    Two legacy Case 2 moment pairs give q=1.0000000000000002.
    Legacy Mix clamps q within 1e-6 of [0, 1]; DynaPlex 1.14 rejects it.
    Build the selected endpoint component with the same legacy parameter p.
    The requested moments, feature values, and all ordinary fits stay unchanged.
    """
    if mean > 0.0 and std >= 0.0:
        cv = std / mean
        a = cv * cv - 1.0 / mean
        if abs(a) >= 1e-6 and a < 0.0 and abs(a + 1.0) >= 1e-10:
            k = int(math.floor(1.0 / -a))
            radicand = -a * k * (1.0 + k) - k
            if radicand >= 0.0:
                q = (1.0 + a * (1.0 + k) + math.sqrt(radicand)) / (1.0 + a)
                p = mean / (k + 1.0 - q)
                if q > 1.0 and q < 1.0 + 1e-6:
                    return DiscreteDist.binomial(k, p)
                if q < 0.0 and q > -1e-6:
                    return DiscreteDist.binomial(k + 1, p)
        elif a >= 1e-6 and a < 1.0:
            k = int(math.floor(1.0 / a))
            radicand = (1.0 + k) * (1.0 - a * k)
            if radicand >= 0.0:
                q = ((1.0 + k) * a - math.sqrt(radicand)) / (1.0 + a)
                p = 1.0 - mean / (k + 1.0 - q + mean)
                if q > 1.0 and q < 1.0 + 1e-6:
                    return DiscreteDist.negative_binomial(float(k), p)
                if q < 0.0 and q > -1e-6:
                    return DiscreteDist.negative_binomial(float(k + 1), p)
    return DiscreteDist.adan_eenige_resing(mean, std)
