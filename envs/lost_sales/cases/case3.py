"""Case 3: stochastic lead times, without and with order crossover (240 instances)."""

from .instance import DEMAND_PATTERNS, Instance, instance

STOCHASTIC_MEANS = (
    (5.0,),
    (8.0, 10.0, 6.0),
    (11.0, 3.0, 5.0, 8.0, 6.0),
    (3.0, 5.0, 5.0, 7.0, 7.0, 10.0, 10.0),
)
LEADTIME_DISTRIBUTIONS = (
    (0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
    (0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.3, 0.1, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2),
    (0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.1, 0.1, 0.15, 0.25, 0.3),
    (0.05, 0.05, 0.1, 0.1, 0.15, 0.25, 0.3, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.1, 0.15, 0.25, 0.25, 0.15, 0.1, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5),
    (0.0, 0.0, 0.0, 0.3, 0.4, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0),
)


def instances() -> list[Instance]:
    result = []
    for crossover in (False, True):
        for probs in LEADTIME_DISTRIBUTIONS:
            for means in STOCHASTIC_MEANS:
                for penalty in (9.0, 39.0, 69.0):
                    families = DEMAND_PATTERNS["mix"][: len(means)]
                    modes = ("known", "demand", "leadtime", "both")
                    arrival = {
                        "stochastic_leadtime": True,
                        "order_crossover": crossover,
                        "leadtime_distribution": list(probs),
                    }
                    result.append(instance("case3", len(result), means, families, penalty, modes, **arrival))
    return result
