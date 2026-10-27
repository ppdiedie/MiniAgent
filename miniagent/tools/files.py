"""文件工具：所有读写都被限制在指定的沙箱目录内。

模型给出的路径属于**不可信输入**，必须防住：
    - 路径穿越：``../../etc/passwd``、``..\\..\\windows\\system32``
    - 绝对路径逃逸：``C:\\Users\\...``、``/etc/hosts``
    - 符号链接指向沙箱外（用 resolve 后的真实路径再比一次）
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .base import tool

_MAX_READ_BYTES = 200_000
_TEXT_SUFFIXES = {".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv", ".log", ".html", ".toml", ".ini"}
# Windows 盘符路径（C:/... 或 C:\...）。Path.is_absolute() 只在 Windows 上认得它，
# 在 POSIX 上会被当成相对路径放行，因此需要单独匹配。
_WIN_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def _sandbox_root() -> Path:
    """沙箱根目录，默认是项目下的 .workspace/，可通过环境变量覆盖。"""
    root = os.getenv("MINIAGENT_SANDBOX")
    path = Path(root) if root else Path.cwd() / ".workspace"
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _resolve(path_text: str) -> Path:
    """把用户路径解析成沙箱内的真实路径，越界直接拒绝。"""
    if not path_text or not path_text.strip():
        raise ValueError("路径不能为空")

    root = _sandbox_root()
    raw = path_text.strip().replace("\\", "/")
    candidate = Path(raw)
    if candidate.is_absolute() or candidate.drive or _WIN_ABSOLUTE.match(raw):
        raise ValueError("不允许使用绝对路径，请给出相对于工作区的路径")

    target = (root / candidate).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise ValueError(f"路径越界：{path_text} 不在工作区内") from None
    return target


@tool(description="读取工作区内文本文件的内容")
def read_file(path: str) -> str:
    """path: 相对于工作区的文件路径，例如 notes/todo.md"""
    target = _resolve(path)
    if not target.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    if not target.is_file():
        raise ValueError(f"不是文件：{path}")
    if target.suffix.lower() not in _TEXT_SUFFIXES:
        raise ValueError(f"只支持文本文件，当前为 {target.suffix or '无后缀'}")

    data = target.read_bytes()[:_MAX_READ_BYTES]
    text = data.decode("utf-8", "replace")
    if target.stat().st_size > _MAX_READ_BYTES:
        text += f"\n...(已截断，文件共 {target.stat().st_size} 字节)"
    return text


@tool(description="在工作区内新建或覆盖写入文本文件")
def write_file(path: str, content: str) -> str:
    """path: 相对于工作区的文件路径
    content: 要写入的完整文本内容"""
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"已写入 {target.relative_to(_sandbox_root())}（{len(content)} 字符）"


@tool(description="列出工作区某个目录下的文件与子目录")
def list_dir(path: str = ".") -> str:
    """path: 相对于工作区的目录，默认为工作区根目录"""
    target = _resolve(path)
    if not target.exists():
        raise FileNotFoundError(f"目录不存在：{path}")
    if not target.is_dir():
        raise ValueError(f"不是目录：{path}")

    entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
    if not entries:
        return f"{path} 是空目录"

    lines = []
    for entry in entries:
        if entry.is_dir():
            lines.append(f"[DIR ] {entry.name}/")
        else:
            lines.append(f"[FILE] {entry.name} ({entry.stat().st_size} B)")
    return "\n".join(lines)
