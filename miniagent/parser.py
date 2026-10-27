"""ReAct 输出解析器：把模型返回的纯文本解析成结构化步骤。

这是整个项目里最容易出 bug、也最能体现工程功力的地方——模型输出不可控，
所以解析必须「宽容但不含糊」：容忍 markdown 代码块、全角冒号、单引号 JSON、
尾随逗号、裸值参数，同时对无法识别的内容给出明确结论而不是静默丢弃。

约定的输出格式::

    Thought: 我需要先算出结果
    Action: calculator
    Action Input: {"expression": "(12+8)*3"}

    （运行时注入 Observation，模型在下一步能看到）

    Thought: 已经拿到结果了
    Final Answer: 结果是 60
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n?(.*?)```", re.DOTALL)
_FINAL_RE = re.compile(r"final\s*answer\s*[:：]", re.IGNORECASE)
_ACTION_RE = re.compile(r"^[ \t]*(?:action|工具)\s*[:：][ \t]*(.+?)[ \t]*$", re.IGNORECASE | re.MULTILINE)
_INPUT_RE = re.compile(
    r"^[ \t]*(?:action[ \t]*input|工具参数)\s*[:：][ \t]*(.+)$",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
_THOUGHT_RE = re.compile(
    r"^[ \t]*(?:thought|思考)\s*[:：][ \t]*(.*?)(?=^[ \t]*(?:action[ \t]*input|action|final\s*answer|observation|工具|思考)\s*[:：]|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
_OBS_TAIL_RE = re.compile(r"\n[ \t]*(?:observation|观察)\s*[:：]", re.IGNORECASE)

# 模型偶尔会把工具名写成 "calculator(expression=...)" 这种形式
_ACTION_CALL_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", re.DOTALL)


@dataclass(slots=True)
class ParsedStep:
    """一次模型输出的解析结果。"""

    thought: str = ""
    action: str | None = None
    action_input: Any = field(default=None)
    final_answer: str | None = None
    raw: str = ""

    @property
    def kind(self) -> str:
        if self.final_answer is not None:
            return "final"
        if self.action:
            return "action"
        return "empty"

    def __str__(self) -> str:
        if self.kind == "final":
            return f"<final> {self.final_answer[:60]}"
        if self.kind == "action":
            return f"<action> {self.action}({self.action_input!r})"
        return "<empty>"


def parse(text: str) -> ParsedStep:
    """解析模型输出。永不抛异常——解析失败时返回 kind == "empty"。"""
    raw = text or ""
    cleaned = _strip_fences(raw).strip()
    if not cleaned:
        return ParsedStep(raw=raw)

    thought = _extract_thought(cleaned)

    # 1) 优先级最高：Final Answer
    final_match = _FINAL_RE.search(cleaned)
    if final_match:
        body = cleaned[final_match.end() :]
        body = _OBS_TAIL_RE.split(body)[0]
        return ParsedStep(thought=thought, final_answer=body.strip(), raw=raw)

    # 2) 其次：Action + Action Input
    action_match = _ACTION_RE.search(cleaned)
    if action_match:
        action_name, inline_args = _split_inline_args(action_match.group(1))
        raw_input = inline_args if inline_args is not None else _extract_input_block(cleaned)
        return ParsedStep(
            thought=thought,
            action=action_name,
            action_input=_parse_args(raw_input),
            raw=raw,
        )

    # 3) 兜底
    if thought:
        # 有 Thought 但既没有 Action 也没有 Final Answer —— 协议没走完。
        # 不能把思考过程当成答案返回给用户，判为无效输出，交给上层要求重试。
        return ParsedStep(thought=thought, raw=raw)

    # 完全没有 ReAct 标签的纯自然语言，视为直接回答（对弱模型更宽容）
    return ParsedStep(thought=thought, final_answer=cleaned, raw=raw)


def _strip_fences(text: str) -> str:
    """展开 markdown 代码块，保留内部内容。"""
    if "```" not in text:
        return text
    blocks = _FENCE_RE.findall(text)
    if not blocks:
        return text.replace("```", "")
    # 若代码块外还有正文（如 Thought），拼接起来以免丢失信息
    outside = _FENCE_RE.sub("", text).strip()
    inside = "\n".join(b.strip() for b in blocks)
    return f"{outside}\n{inside}".strip() if outside else inside


def _extract_thought(text: str) -> str:
    m = _THOUGHT_RE.search(text)
    return m.group(1).strip() if m else ""


def _split_inline_args(action_raw: str) -> tuple[str, str | None]:
    """处理 `calculator({"expression": "1+1"})` 这类写法。"""
    name = action_raw.strip().strip("`").strip()
    m = _ACTION_CALL_RE.match(name)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # 去掉可能粘在后面的标点
    return name.rstrip("。.,;").strip(), None


def _extract_input_block(text: str) -> str:
    m = _INPUT_RE.search(text)
    if not m:
        return ""
    block = m.group(1).strip()
    block = _OBS_TAIL_RE.split(block)[0].strip()
    return block


def _parse_args(raw: str) -> Any:
    """尽最大努力把参数文本转成 dict；转不了就原样返回字符串。"""
    if raw is None:
        return {}
    raw = raw.strip().strip("`").strip()
    if not raw:
        return {}

    # 去掉模型偶尔加的 Observation 回显
    raw = _OBS_TAIL_RE.split(raw)[0].strip()
    if not raw:
        return {}

    for candidate in _json_candidates(raw):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
        # 模型只给了一个裸值，例如 calculator("1+1") 中的 "1+1"
        return {"__value__": value}

    # JSON 全线失败：交给运行时按「单参数工具」兜底处理
    return raw


def _json_candidates(raw: str) -> list[str]:
    """生成若干可能的 JSON 文本变体，容忍常见的小模型坏习惯。"""
    candidates = [raw]

    # 去掉外层多余括号前的说明文字，如 "参数是 {"a": 1}"
    brace = raw.find("{")
    if brace > 0:
        candidates.append(raw[brace:])

    base = candidates[-1]
    if "'" in base:  # 单引号 → 双引号
        candidates.append(base.replace("'", '"'))
    if re.search(r",\s*[}\]]", base):  # 尾随逗号
        candidates.append(re.sub(r",\s*([}\]])", r"\1", base))
        if "'" in base:
            sq = base.replace("'", '"')
            candidates.append(re.sub(r",\s*([}\]])", r"\1", sq))

    return candidates
