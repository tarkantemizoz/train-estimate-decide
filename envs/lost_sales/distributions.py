"""The distributions both lost-sales models simulate with, so that they draw the same numbers.

fit_aer gives the demand distribution of a mean and standard deviation, and arriving_units the arrivals of one
age in transit under order crossover, from a single random draw.
"""

import math

from dynaplex.modelling import DiscreteDist


def fit_aer(mean: float, std: float) -> DiscreteDist:
    """DynaPlex's native AER fit, except where legacy Mix clamps an endpoint.

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


def arriving_units(units: int, probability: float, draw: float) -> int:
    """The number of units arriving when each arrives independently with this probability.

    The count is binomial: the quantile of Binomial(units, probability) at the uniform draw. Units of one
    age are interchangeable, so this is the same distribution as one draw per unit. One draw per age,
    whatever the orders, keeps every policy on the same random numbers, so that comparisons stay paired.
    """
    if units == 0 or probability <= 0.0:
        return 0
    if probability >= 1.0:
        return units
    # Count the less likely outcome, arrivals or not, so that the first probability cannot underflow.
    flip = probability > 0.5
    q = 1.0 - probability if flip else probability
    level = 1.0 - draw if flip else draw
    ratio = q / (1.0 - q)
    mass = (1.0 - q) ** units
    cumulative = mass
    count = 0
    while cumulative < level and count < units:
        mass *= ratio * (units - count) / (count + 1)
        count += 1
        cumulative += mass
    return units - count if flip else count
