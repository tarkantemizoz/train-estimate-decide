"""Case 2: cyclic demand with cycles of 3, 5 and 7 periods (243 instances)."""

from .instance import DEMAND_PATTERNS, Instance, instance

CYCLIC_MEANS = (
    ((2.5, 4.5, 3.0), (9.0, 11.0, 9.5), (3.0, 6.0, 10.0)),
    ((4.0, 2.5, 3.0, 4.5, 3.0), (7.5, 11.5, 10.0, 9.5, 8.5), (11.0, 3.0, 5.0, 8.0, 6.0)),
    (
        (3.0, 4.0, 2.0, 3.5, 4.5, 2.5, 4.0),
        (11.0, 10.0, 9.0, 10.0, 11.0, 8.5, 9.5),
        (3.0, 5.0, 5.0, 7.0, 7.0, 10.0, 10.0),
    ),
)


def instances() -> list[Instance]:
    result = []
    for mean_group in CYCLIC_MEANS:
        for means in mean_group:
            for families in DEMAND_PATTERNS.values():
                for penalty in (9.0, 39.0, 69.0):
                    for leadtime in (3, 6, 9):
                        modes = ("known", "demand")
                        result.append(
                            instance(
                                "case2",
                                len(result),
                                means,
                                families[: len(means)],
                                penalty,
                                modes,
                                leadtime=leadtime,
                            )
                        )
    return result
