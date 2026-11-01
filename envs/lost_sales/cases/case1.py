"""Case 1: stationary demand and deterministic lead times (320 instances)."""

from .instance import Instance, instance


def instances() -> list[Instance]:
    result = []
    for penalty in (9.0, 39.0, 69.0, 99.0):
        for mean in (3.0, 5.0, 7.0, 10.0):
            for family in ("binomial", "poisson", "negative_binomial", "geometric"):
                for leadtime in (2, 4, 6, 8, 10):
                    modes = ("known", "demand")
                    result.append(
                        instance("case1", len(result), (mean,), (family,), penalty, modes, leadtime=leadtime)
                    )
    return result
