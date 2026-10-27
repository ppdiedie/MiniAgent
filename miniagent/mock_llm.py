"""离线 Mock 模型：用规则引擎模拟一个会使用工具的 LLM。

存在意义（也是简历里值得讲的一点）：
    Agent 框架的绝大部分 bug 出在「循环调度、解析、工具执行、记忆裁剪」，
    而不是模型本身。有了 MockLLM，这些逻辑可以用单元测试完全覆盖，
    不花一分钱、不依赖网络，CI 里也能稳定跑。

它按 ReAct 协议输出 Thought / Action / Action Input，并会读取真实注入的
Observation 来生成 Final Answer，因此走的是和真实模型**完全相同的代码路径**。
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Callable, Sequence

from .llm import Message

_OBS_PREFIX = re.compile(r"^\s*(?:Observation|观察)\s*[:：]\s*", re.IGNORECASE)
_CN_BRACKETS = str.maketrans({"（": "(", "）": ")", "×": "*", "÷": "/", "　": " "})
_SAFE_EXPR_CHARS = re.compile(r"[-+*/%().\d\s]{3,}")


@dataclass(slots=True)
class _Action:
    thought: str
    tool: str
    params: dict


class MockLLM:
    """确定性的规则模型，接口与真实 LLM 一致。"""

    model_name = "mock-rule-engine"

    def __init__(self, latency: float = 0.0) -> None:
        self.latency = latency
        self.calls = 0

    # ------------------------------------------------------------------ 对外接口
    def chat(self, messages: Sequence[Message], stop: Sequence[str] | None = None) -> str:
        self.calls += 1
        question = _current_question(messages)
        observations = _collect_observations(messages)

        plan, finalizer = _build_plan(question)
        index = len(observations)

        if index < len(plan):
            action = plan[index]
            payload = json.dumps(action.params, ensure_ascii=False)
            return (
                f"Thought: {action.thought}\n"
                f"Action: {action.tool}\n"
                f"Action Input: {payload}"
            )

        return f"Thought: 我已经拿到需要的信息。\nFinal Answer: {finalizer(observations)}"


# --------------------------------------------------------------------------- 消息解析
def _current_question(messages: Sequence[Message]) -> str:
    for msg in reversed(messages):
        if msg.role == "user" and not _OBS_PREFIX.match(msg.content):
            return msg.content.strip()
    return ""


def _collect_observations(messages: Sequence[Message]) -> list[str]:
    """只收集**本轮任务**产生的观察，避免多轮对话时历史 observation 干扰计数。"""
    question_index = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.role == "user" and not _OBS_PREFIX.match(msg.content):
            question_index = i
            break

    return [
        _OBS_PREFIX.sub("", m.content).strip()
        for m in messages[question_index + 1 :]
        if m.role == "user" and _OBS_PREFIX.match(m.content)
    ]


# --------------------------------------------------------------------------- 意图识别
def _build_plan(question: str) -> tuple[list[_Action], Callable[[list[str]], str]]:
    lowered = question.lower()
    actions: list[_Action] = []
    labels: list[str] = []

    wants_time = any(k in question for k in ("几点", "现在时间", "当前时间", "今天日期", "几号"))
    wants_todo = any(k in question for k in ("待办", "todo", "提醒我", "记一下", "记个"))
    wants_list_todo = any(k in question for k in ("列表", "列出", "有哪些", "查看待办", "看看待办"))
    wants_file = any(k in question for k in ("文件", "目录", "文件夹"))
    expression = _extract_expression(question)

    # 1) 计算
    if expression:
        actions.append(_Action("用户给出了数学表达式，我用 calculator 精确求值，避免心算出错。", "calculator", {"expression": expression}))
        labels.append("calc")

    # 2) 时间
    if wants_time:
        actions.append(_Action("需要当前时间才能正确理解「今天/现在」这类相对表述。", "now", {}))
        labels.append("time")

    # 3) 待办
    if wants_todo:
        title = _extract_todo_title(question)
        if wants_list_todo or not title:
            actions.append(_Action("查一下现有待办，避免重复添加。", "todo_list", {}))
            labels.append("todo_list")
        else:
            actions.append(_Action("把这件要做的事登记到待办清单里。", "todo_add", {"title": title}))
            labels.append("todo_add")

    # 4) 文件
    elif wants_file:
        path = _extract_path(question)
        if any(k in question for k in ("读", "内容", "打开")) and path:
            actions.append(_Action("直接读取目标文件内容。", "read_file", {"path": path}))
            labels.append("read_file")
        else:
            actions.append(_Action("先列出工作区目录，确认有哪些文件可用。", "list_dir", {"path": path or "."}))
            labels.append("list_dir")

    if not actions:
        return [], lambda _obs: _fallback_answer(question)

    return actions, _make_finalizer(labels, question)


# --------------------------------------------------------------------------- 收尾生成
def _make_finalizer(labels: list[str], question: str) -> Callable[[list[str]], str]:
    def finalize(observations: list[str]) -> str:
        parts: list[str] = []
        for label, obs in zip(labels, observations):
            failed = obs.startswith("工具执行失败")
            if label == "calc":
                parts.append(
                    f"计算结果：{obs}" if not failed else f"计算没能完成（{obs.split('：', 1)[-1]}），请检查表达式。"
                )
            elif label == "time":
                parts.append(f"当前时间是 {obs}。")
            elif label == "todo_add":
                parts.append(f"待办已记录：{obs}")
            elif label == "todo_list":
                parts.append(f"待办清单如下：\n{obs}")
            elif label == "read_file":
                preview = obs if len(obs) <= 300 else obs[:300] + "..."
                parts.append(f"文件内容：\n{preview}" if not failed else f"读取失败：{obs}")
            else:
                parts.append(f"工作区文件：\n{obs}" if not failed else f"列目录失败：{obs}")

        if len(observations) < len(labels):
            parts.append("（部分步骤未完成）")

        return "\n".join(parts)

    return finalize


def _fallback_answer(question: str) -> str:
    return (
        "当前运行在离线 Mock 模式（未配置真实模型），我只能演示预设的几类任务。\n"
        "已支持：数学计算、查询时间、添加/查看待办、读取/列出工作区文件。\n"
        f"你的输入是「{question[:40]}」，没有匹配到可用的工具组合。\n"
        "配置 OPENAI_API_KEY 后即可切换到真实模型进行开放式对话。"
    )


# --------------------------------------------------------------------------- 文本抽取
def _extract_expression(text: str) -> str | None:
    """从中文自然语言里抠出可求值的数学表达式。"""
    normalized = text.translate(_CN_BRACKETS)
    for chunk in _SAFE_EXPR_CHARS.findall(normalized):
        candidate = chunk.strip().strip(".")
        candidate = candidate.strip("+-*/% ")
        if len(candidate) < 2:
            continue
        if not any(op in candidate for op in "+-*/%"):
            continue
        if not re.search(r"\d", candidate):
            continue
        if candidate.count("(") != candidate.count(")"):
            continue
        try:
            ast.parse(candidate, mode="eval")
        except SyntaxError:
            continue
        return candidate
    return None


def _extract_todo_title(text: str) -> str | None:
    patterns = [
        r"待办[：:是为]?\s*(.+)",
        r"待办事项[：:是为]?\s*(.+)",
        r"提醒我\s*(.+)",
        r"记一下[：:，,]?\s*(.+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            title = m.group(1).strip(" 。.，,；;")
            # 去掉「算一下 xx 然后记个待办：yyy」里的前置描述
            title = re.split(r"[，,；;]\s*(?:然后|并且|再)", title)[-1].strip()
            if title:
                return title[:80]
    return None


def _extract_path(text: str) -> str | None:
    """抽取路径。字符集严格限定为 ASCII，避免把中文描述误当成目录名。"""
    # 1) 带已知后缀的文件名，如 notes/todo.md
    m = re.search(r"[A-Za-z0-9_./\\-]+\.(?:txt|md|py|json|yaml|yml|csv|log)", text)
    if m:
        return m.group(0)
    # 2) 引号包裹的路径
    m = re.search(r"[「\"'`]([A-Za-z0-9_./\\-]+)[」\"'`]", text)
    if m:
        return m.group(1)
    # 3) 「目录 xxx」后面的纯 ASCII 路径
    m = re.search(r"(?:目录|文件夹|工作区)\s*([A-Za-z0-9_./\\-]+)", text)
    return m.group(1) if m else None
