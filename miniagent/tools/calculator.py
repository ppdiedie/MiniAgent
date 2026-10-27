"""计算器工具：用 AST 白名单求值，彻底避开 ``eval`` 的代码注入风险。

为什么不用 eval：
    eval("__import__('os').system('rm -rf /')") —— 模型输出直接进 eval 等于把
    执行权交给一个不可信来源。这里只允许数字、四则运算、幂、少量数学函数，
    并对指数规模设上限，顺手防住 ``9**9**9`` 这类 CPU 炸弹。
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from .base import tool

_MAX_EXPONENT = 1000
_MAX_NODES = 200

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_SAFE_FUNCS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
}
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


@tool(description="计算数学表达式，支持 + - * / // % ** 与 sqrt/log/sin 等函数")
def calculator(expression: str) -> str:
    """expression: 需要求值的数学表达式，例如 "(12+8)*3/5\""""
    expr = expression.strip()
    if not expr:
        raise ValueError("表达式为空")
    if len(expr) > 500:
        raise ValueError("表达式过长（上限 500 字符）")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"表达式语法错误：{exc.msg}") from exc

    if sum(1 for _ in ast.walk(tree)) > _MAX_NODES:
        raise ValueError("表达式过于复杂")

    result = _eval_node(tree.body)
    return _format_number(result)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        raise ValueError(f"不允许的常量类型：{type(node.value).__name__}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise ValueError(f"不支持的运算符：{op_type.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if op_type is ast.Pow and abs(right) > _MAX_EXPONENT:
            raise ValueError(f"指数过大（上限 {_MAX_EXPONENT}）")
        if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ValueError("除数不能为 0")
        return _BIN_OPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"不支持的一元运算符：{op_type.__name__}")
        return _UNARY_OPS[op_type](_eval_node(node.operand))

    if isinstance(node, ast.Name):
        if node.id in _CONSTS:
            return float(_CONSTS[node.id])
        raise ValueError(f"未知标识符：{node.id}")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("不支持链式调用")
        fname = node.func.id
        if fname not in _SAFE_FUNCS:
            raise ValueError(f"不允许调用函数：{fname}")
        if node.keywords:
            raise ValueError("不支持关键字参数")
        args = [_eval_node(a) for a in node.args]
        return float(_SAFE_FUNCS[fname](*args))

    raise ValueError(f"不支持的语法节点：{type(node).__name__}")


def _format_number(value: float) -> str:
    """整数不显示小数点，浮点保留 6 位有效小数，避免 0.30000000000000004 这种噪声。"""
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("计算结果不是有限数值")
    if float(value).is_integer() and abs(value) < 1e16:
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")
