import pytest

from rag.tools import CalculatorError, calculate, format_number


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("1.5 * 4", 6.0),
        ("2 + 3 * 4", 14.0),  # usual precedence
        ("(2 + 3) * 4", 20.0),
        ("-3 + 5", 2.0),
        ("10 / 4", 2.5),
        ("2 ** 3", 8.0),
        ("2^3", 8.0),  # written the engineer's way
        ("7 % 4", 3.0),
        ("0.5 * 18.5 * 2.0 ** 2", 37.0),
    ],
)
def test_arithmetic(expression, expected):
    assert calculate(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('ls')",  # function call
        "abs(-1)",
        "x + 1",  # names
        "(1).real",  # attribute access
        "[1, 2]",
        "'1' + '2'",
        "True + 1",
        "1 < 2",
    ],
)
def test_anything_but_arithmetic_is_rejected(expression):
    with pytest.raises(CalculatorError):
        calculate(expression)


@pytest.mark.parametrize(
    "expression",
    ["1 / 0", "9 ** 9 ** 9", "10.0 ** 400", "(-8) ** 0.5", "1e308 * 10", "1 +", "1" * 201],
)
def test_impossible_or_oversized_calculations_give_an_error_not_a_crash(expression):
    with pytest.raises(CalculatorError):
        calculate(expression)


def test_results_are_formatted_without_float_noise():
    assert format_number(calculate("0.1 + 0.2")) == "0.3"
    assert format_number(calculate("1 / 3")) == "0.3333333333"
    assert format_number(calculate("2 * 3")) == "6"
