"""工具包。

导入本包会自动完成所有内置工具的注册（各模块在 import 时执行 ``@tool`` 装饰器）。
新增工具只需：写一个模块 → 在里面用 ``@tool`` 装饰函数 → 在下方 import 一次。
"""

from .base import (
    ToolError,
    ToolResult,
    ToolSpec,
    build_tool_prompt,
    clear_registry,
    execute_tool,
    get_tool,
    list_tools,
    tool,
)

# 以下 import 的副作用即为「注册内置工具」，顺序无关
from .calculator import calculator
from .clock import now
from .files import list_dir, read_file, write_file
from .todo import todo_add, todo_done, todo_list

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
    # 内置工具
    "calculator",
    "now",
    "read_file",
    "write_file",
    "list_dir",
    "todo_add",
    "todo_done",
    "todo_list",
]
