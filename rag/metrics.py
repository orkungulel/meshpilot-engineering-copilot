"""Statistics for a small evaluation set.

With 30 questions a single percentage hides a lot of uncertainty, so every rate
is reported with a 95% Wilson interval, and two configurations are compared
with an exact McNemar test on the questions where they disagree. How well the
judge agrees with a person's labels is also given as Cohen's kappa.
"""

import math


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion.

    Unlike the textbook normal approximation (p ± z·√(p(1-p)/n)), it stays
    inside [0, 1] and behaves sensibly for small n and for rates near 0 or 1.
    """
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError(f"need 0 <= successes <= n and n > 0, got {successes=}, {n=}")
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def mcnemar_exact_p(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar test for two configurations run on the same questions.

    Questions both get right (or both get wrong) say nothing about which is
    better. If neither is better, each disagreement is a fair coin flip, so the
    p-value is the binomial probability of a split at least this uneven.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


def cohens_kappa(both_yes: int, only_a_yes: int, only_b_yes: int, both_no: int) -> float | None:
    """Agreement between two graders beyond what chance would give: 1 is perfect, 0 is chance level.

    Raw agreement flatters a judge when most answers are correct: a judge that
    passes everything agrees with a person on every correct answer. Kappa
    subtracts the agreement two graders with the same pass rates would reach
    by chance. It is None when undefined: both gave every answer the same grade.
    """
    n = both_yes + only_a_yes + only_b_yes + both_no
    if n <= 0:
        raise ValueError("need at least one graded answer")
    observed = (both_yes + both_no) / n
    a_yes = (both_yes + only_a_yes) / n
    b_yes = (both_yes + only_b_yes) / n
    chance = a_yes * b_yes + (1 - a_yes) * (1 - b_yes)
    if chance == 1:
        return None
    return (observed - chance) / (1 - chance)
