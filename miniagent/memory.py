"""对话记忆。

Agent 的上下文是有限资源，长任务很容易把窗口撑爆。这里做一个**双层裁剪**：

1. 轮数裁剪：只保留最近 ``max_turns`` 条消息。
2. 字符预算裁剪：估算总长度，超预算时继续从头丢弃，并插入一条省略提示。

裁剪时始终保证「首个非系统消息是 user」，否则部分模型会拒绝请求。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .llm import Message

_ELLIPSIS_MARK = "[上下文已裁剪，省略了 {n} 条较早消息]"


@dataclass
class ConversationMemory:
    max_turns: int = 8
    max_chars: int = 12_000
    _items: list[Message] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------ 写入
    def add(self, message: Message) -> None:
        self._items.append(message)

    def add_user(self, content: str) -> None:
        self.add(Message("user", content))

    def add_assistant(self, content: str) -> None:
        self.add(Message("assistant", content))

    # ------------------------------------------------------------------ 读取
    def messages(self) -> list[Message]:
        """返回裁剪后的消息列表（不含 system，由 Agent 自行拼装）。"""
        items = self._items

        # 第一层：轮数裁剪
        if len(items) > self.max_turns:
            dropped = len(items) - self.max_turns
            items = [Message("user", _ELLIPSIS_MARK.format(n=dropped))] + items[-self.max_turns :]
        else:
            items = list(items)

        # 第二层：字符预算裁剪
        items = self._trim_by_chars(items)

        return items

    def _trim_by_chars(self, items: list[Message]) -> list[Message]:
        total = sum(len(m.content) for m in items)
        if total <= self.max_chars:
            return items

        dropped = 0
        while items and total > self.max_chars and len(items) > 2:
            removed = items.pop(0)
            total -= len(removed.content)
            dropped += 1
            # 跳过被丢成 assistant 开头的局面
            while items and items[0].role != "user" and len(items) > 1:
                removed = items.pop(0)
                total -= len(removed.content)
                dropped += 1

        if dropped:
            items.insert(0, Message("user", _ELLIPSIS_MARK.format(n=dropped)))
        return items

    # ------------------------------------------------------------------ 维护
    def extend(self, messages: Iterable[Message]) -> None:
        for m in messages:
            self.add(m)

    def reset(self) -> None:
        self._items.clear()

    @property
    def raw(self) -> list[Message]:
        """未裁剪的完整历史，便于调试与导出。"""
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)
