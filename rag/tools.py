"""The agent's calculator: arithmetic on a parsed expression, never eval().

The expression is parsed into a Python syntax tree and evaluated node by node.
Only numbers, + - * / % **, unary signs and parentheses are allowed; names,
function calls and attribute access are rejected, so model output can never
run code.
"""

import ast
import math
import operator

MAX_EXPRESSION_LENGTH = 200

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class CalculatorError(ValueError):
    pass


def calculate(expression: str) -> float:
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise CalculatorError(f"expression longer than {MAX_EXPRESSION_LENGTH} characters")
    try:
        # Engineers write powers as 2^3; in Python ^ would be a bitwise XOR.
        tree = ast.parse(expression.replace("^", "**"), mode="eval")
    except SyntaxError as exc:
        raise CalculatorError("not a valid arithmetic expression") from exc
    result = _evaluate(tree.body)
    if isinstance(result, complex) or not math.isfinite(result):
        raise CalculatorError("the result is not a finite real number")
    return result


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        # Floats overflow quickly; Python integers would grow without limit (9**9**9).
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        try:
            return OPERATORS[type(node.op)](left, right)
        except (ZeroDivisionError, OverflowError) as exc:
            raise CalculatorError(str(exc)) from exc
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](_evaluate(node.operand))
    raise CalculatorError(f"only numbers and + - * / % ** ( ) are allowed, not {type(node).__name__}")


def format_number(value: float) -> str:
    """Ten significant digits, without float noise such as 0.30000000000000004."""
    return format(value, ".10g")
