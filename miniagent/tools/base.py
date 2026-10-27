"""工具系统。

三个目标：
1. **零样板** —— 一个 ``@tool`` 装饰器即可完成注册，JSON Schema 由类型注解自动生成。
2. **值得信任** —— 参数在真正执行前做类型与必填校验，工具内部异常不炸掉 Agent 循环。
3. **可观测** —— 每次调用都产出结构化结果（耗时、成功与否、错误原因）。

用法::

    @tool(description="把两个数相加")
    def add(a: int, b: int) -> int:
        return a + b
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable, get_args, get_origin, get_type_hints

__all__ = [
    "ToolSpec",
    "ToolResult",
    "ToolError",
    "tool",
    "get_tool",
    "list_tools",
    "clear_registry",
    "build_tool_prompt",
    "execute_tool",
]


class ToolError(ValueError):
    """参数不合法或工具不存在。"""


# --------------------------------------------------------------------------- 数据结构
@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]

    def to_openai_schema(self) -> dict[str, Any]:
        """导出为 OpenAI function-calling 格式，方便迁移到原生 tool_calls。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(slots=True)
class ToolResult:
    ok: bool
    content: str
    tool: str = ""
    elapsed_ms: float = 0.0
    error: str | None = None
    payload: Any = field(default=None)

    @classmethod
    def failure(cls, tool: str, error: str, elapsed_ms: float = 0.0) -> ToolResult:
        return cls(ok=False, content=f"工具执行失败：{error}", tool=tool, elapsed_ms=elapsed_ms, error=error)

    def __str__(self) -> str:
        return self.content


# --------------------------------------------------------------------------- 注册表
_REGISTRY: dict[str, ToolSpec] = {}

_PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean"}


def tool(
    _func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[..., Any]:
    """注册工具。既支持 ``@tool`` 也支持 ``@tool(description=...)``。"""

    def register(func: Callable[..., Any]) -> Callable[..., Any]:
        spec = _build_spec(func, name=name, description=description)
        if spec.name in _REGISTRY:
            raise ToolError(f"工具名重复注册：{spec.name}")
        _REGISTRY[spec.name] = spec
        return func

    if _func is not None:
        return register(_func)
    return register


def _build_spec(func: Callable[..., Any], *, name: str | None, description: str | None) -> ToolSpec:
    hints = get_type_hints(func)
    sig = inspect.signature(func)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in {"self", "cls"}:
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue

        annotation = hints.get(param_name, str)
        schema, nullable = _annotation_to_schema(annotation)
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
        else:
            schema["default"] = param.default
        if nullable:
            schema["nullable"] = True

        doc = _param_doc(func, param_name)
        if doc:
            schema["description"] = doc
        properties[param_name] = schema

    return ToolSpec(
        name=name or func.__name__,
        description=(description or (func.__doc__ or "").strip().splitlines()[0] if (description or func.__doc__) else func.__name__),
        parameters={
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        func=func,
    )


def _annotation_to_schema(annotation: Any) -> tuple[dict[str, Any], bool]:
    """把 Python 类型注解转成 JSON Schema 片段，返回 (schema, 是否可空)。"""
    origin = get_origin(annotation)

    if origin is not None:
        args = [a for a in get_args(annotation) if a is not type(None)]  # noqa: E721
        nullable = len(args) != len(get_args(annotation))

        if str(origin).endswith("Literal"):
            values = list(args)
            return {"type": _json_type(values[0]) if values else "string", "enum": values}, nullable
        if origin is list:
            item = _annotation_to_schema(args[0])[0] if args else {"type": "string"}
            return {"type": "array", "items": item}, nullable
        if origin is dict:
            return {"type": "object"}, nullable
        if args:
            schema, inner_null = _annotation_to_schema(args[0])
            return schema, nullable or inner_null
        return {"type": "string"}, nullable

    if annotation is bool or annotation is int or annotation is float or annotation is str:
        return {"type": _PY_TO_JSON[annotation]}, False
    if annotation is list:
        return {"type": "array", "items": {"type": "string"}}, False
    if annotation is dict:
        return {"type": "object"}, False
    return {"type": "string"}, False


def _json_type(value: Any) -> str:
    return _PY_TO_JSON.get(type(value), "string")


def _param_doc(func: Callable[..., Any], param_name: str) -> str:
    """从 docstring 的 ``name: 说明`` 行里提取参数说明（轻量解析，不引第三方库）。"""
    doc = inspect.getdoc(func) or ""
    for line in doc.splitlines():
        line = line.strip()
        if line.startswith(f"{param_name}:") or line.startswith(f"{param_name} :"):
            return line.split(":", 1)[1].strip()
    return ""


# --------------------------------------------------------------------------- 查询与执行
def get_tool(name: str) -> ToolSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY)) or "<空>"
        raise ToolError(f"不存在名为 {name!r} 的工具。可用工具：{available}") from None


def list_tools() -> list[ToolSpec]:
    return sorted(_REGISTRY.values(), key=lambda s: s.name)


def clear_registry() -> None:
    """仅供测试使用。"""
    _REGISTRY.clear()


def build_tool_prompt() -> str:
    """渲染成给模型看的工具清单（ReAct 纯文本协议使用）。"""
    lines: list[str] = []
    for spec in list_tools():
        lines.append(f"- {spec.name}: {spec.description}")
        props: dict[str, Any] = spec.parameters.get("properties", {})
        required = set(spec.parameters.get("required", []))
        if props:
            parts = []
            for pname, pschema in props.items():
                flag = "" if pname in required else "（可选）"
                parts.append(f"{pname}: {pschema.get('type', 'string')}{flag}")
            lines.append(f"    参数：{', '.join(parts)}")
    return "\n".join(lines)


def execute_tool(name: str, arguments: Any) -> ToolResult:
    """执行工具。所有异常在此收口，保证 Agent 主循环永不因工具崩溃而中断。"""
    started = time.perf_counter()

    try:
        spec = get_tool(name)
    except ToolError as exc:
        return ToolResult.failure(name, str(exc))

    try:
        kwargs = _normalize_arguments(spec, arguments)
        _validate(spec, kwargs)
    except ToolError as exc:
        return ToolResult.failure(name, str(exc), (time.perf_counter() - started) * 1000)

    try:
        output = spec.func(**kwargs)
    except Exception as exc:  # noqa: BLE001 - 工具内部任何异常都降级为观察结果
        elapsed = (time.perf_counter() - started) * 1000
        return ToolResult.failure(name, f"{type(exc).__name__}: {exc}", elapsed)

    elapsed = (time.perf_counter() - started) * 1000
    text = output if isinstance(output, str) else _stringify(output)
    return ToolResult(ok=True, content=text, tool=name, elapsed_ms=elapsed, payload=output)


def _stringify(value: Any) -> str:
    import json

    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _normalize_arguments(spec: ToolSpec, arguments: Any) -> dict[str, Any]:
    """把模型给出的各种「不规整」参数整理成关键字参数。"""
    props: dict[str, Any] = spec.parameters.get("properties", {})

    if arguments is None:
        arguments = {}
    if isinstance(arguments, dict):
        if "__value__" in arguments and len(arguments) == 1:
            arguments = arguments["__value__"]
        else:
            return dict(arguments)

    # 走到这里说明模型给的是裸值（字符串/数字），绑定到唯一参数上
    if len(props) == 1:
        only = next(iter(props))
        return {only: arguments}
    if not props:
        return {}
    raise ToolError(
        f"工具 {spec.name} 需要参数 {sorted(props)}，但模型只提供了裸值 {arguments!r}，"
        "请让模型输出 JSON 对象形式的参数。"
    )


def _validate(spec: ToolSpec, kwargs: dict[str, Any]) -> None:
    props: dict[str, Any] = spec.parameters.get("properties", {})
    required: list[str] = spec.parameters.get("required", [])

    missing = [k for k in required if k not in kwargs]
    if missing:
        raise ToolError(f"缺少必填参数：{', '.join(missing)}")

    unknown = [k for k in kwargs if k not in props]
    if unknown:
        raise ToolError(f"未知参数：{', '.join(unknown)}；该工具仅接受 {sorted(props) or '无参数'}")

    for key, value in kwargs.items():
        expected = props[key].get("type")
        if expected and not _type_ok(expected, value):
            raise ToolError(f"参数 {key} 期望 {expected}，实际收到 {type(value).__name__}")


def _type_ok(expected: str, value: Any) -> bool:
    if value is None:  # None 交给工具函数自己决定默认值
        return True
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True
