import pytest

from rag.metrics import cohens_kappa, mcnemar_exact_p, wilson_interval


def test_wilson_interval_matches_a_hand_calculation():
    # 24/30: p = 0.8, centre = 0.86403 / 1.12805, half-width = 0.15681 / 1.12805
    low, high = wilson_interval(24, 30)

    assert low == pytest.approx(0.6269, abs=1e-4)
    assert high == pytest.approx(0.9050, abs=1e-4)


def test_wilson_interval_stays_inside_zero_and_one():
    assert wilson_interval(0, 30)[0] == 0.0
    assert wilson_interval(30, 30)[1] == 1.0
    assert wilson_interval(0, 30)[1] > 0  # zero successes in 30 still leaves real uncertainty


def test_wilson_interval_narrows_with_more_questions():
    low_30, high_30 = wilson_interval(15, 30)
    low_300, high_300 = wilson_interval(150, 300)

    assert high_300 - low_300 < high_30 - low_30


@pytest.mark.parametrize("successes, n", [(1, 0), (-1, 10), (11, 10)])
def test_wilson_interval_rejects_impossible_counts(successes, n):
    with pytest.raises(ValueError):
        wilson_interval(successes, n)


@pytest.mark.parametrize(
    "only_a, only_b, expected",
    [
        (0, 0, 1.0),  # no disagreements: no evidence either way
        (0, 5, 2 / 32),  # 5 of 5 disagreements favour B
        (1, 9, 2 * 11 / 1024),  # P(X <= 1) for X ~ Binomial(10, 0.5), two-sided
        (3, 3, 1.0),
    ],
)
def test_mcnemar_exact_p_values(only_a, only_b, expected):
    assert mcnemar_exact_p(only_a, only_b) == pytest.approx(expected)


def test_mcnemar_is_symmetric():
    assert mcnemar_exact_p(2, 7) == mcnemar_exact_p(7, 2)


def test_cohens_kappa_matches_a_textbook_example():
    # 50 answers: both pass 20, only A passes 5, only B passes 10, both fail 15.
    # Observed agreement 35/50 = 0.7. By chance: 0.5 x 0.6 + 0.5 x 0.4 = 0.5. Kappa: (0.7 - 0.5) / (1 - 0.5).
    assert cohens_kappa(20, 5, 10, 15) == pytest.approx(0.4)


def test_cohens_kappa_is_one_for_perfect_agreement_and_zero_at_chance():
    assert cohens_kappa(20, 0, 0, 10) == pytest.approx(1.0)
    # A grader who passes everything agrees with B on all of B's passes, by construction.
    assert cohens_kappa(25, 5, 0, 0) == pytest.approx(0.0)


def test_cohens_kappa_is_undefined_when_both_give_every_answer_the_same_grade():
    assert cohens_kappa(30, 0, 0, 0) is None
    assert cohens_kappa(0, 0, 0, 30) is None
