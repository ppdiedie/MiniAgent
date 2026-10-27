"""待办工具：用 SQLite 做持久化，演示 Agent 的「跨会话状态」。

工程要点：
    - 所有 SQL 均为参数化查询（``?`` 占位），杜绝拼接字符串导致的注入
    - 连接使用上下文管理器，异常路径也能正确关闭
    - 数据文件默认落在 .workspace/todo.db，测试时可用环境变量重定向
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .base import tool

_SCHEMA = """
CREATE TABLE IF NOT EXISTS todo (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT    NOT NULL,
    done       INTEGER NOT NULL DEFAULT 0,
    created_at TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);
"""


def db_path() -> Path:
    override = os.getenv("MINIAGENT_TODO_DB")
    path = Path(override) if override else Path.cwd() / ".workspace" / "todo.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


@tool(description="新增一条待办事项")
def todo_add(title: str) -> str:
    """title: 待办内容"""
    text = title.strip()
    if not text:
        raise ValueError("待办内容不能为空")
    if len(text) > 200:
        raise ValueError("待办内容过长（上限 200 字）")

    with _connect() as conn:
        cursor = conn.execute("INSERT INTO todo (title) VALUES (?)", (text,))
        return f"已添加待办 #{cursor.lastrowid}：{text}"


@tool(description="把指定编号的待办标记为已完成")
def todo_done(todo_id: int) -> str:
    """todo_id: 待办编号，可用 todo_list 查看"""
    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE todo SET done = 1 WHERE id = ? AND done = 0", (todo_id,)
        )
        if cursor.rowcount == 0:
            exists = conn.execute("SELECT done FROM todo WHERE id = ?", (todo_id,)).fetchone()
            if exists is None:
                raise ValueError(f"待办 #{todo_id} 不存在")
            return f"待办 #{todo_id} 已经是完成状态"
        return f"待办 #{todo_id} 已标记完成"


@tool(description="列出所有待办事项，可选择只看未完成的")
def todo_list(only_pending: bool = False) -> str:
    """only_pending: 为 true 时只返回未完成项"""
    sql = "SELECT id, title, done, created_at FROM todo"
    params: tuple = ()
    if only_pending:
        sql += " WHERE done = 0"
    sql += " ORDER BY done ASC, id ASC"

    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()

    if not rows:
        return "当前没有待办事项"

    lines = []
    for row in rows:
        mark = "[x]" if row["done"] else "[ ]"
        lines.append(f"{mark} #{row['id']} {row['title']}  ({row['created_at']})")

    pending = sum(1 for r in rows if not r["done"])
    lines.append(f"--- 共 {len(rows)} 条，未完成 {pending} 条")
    return "\n".join(lines)
