"""Order crossover: how many units of one age in transit arrive in a period, from a single random draw."""


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
