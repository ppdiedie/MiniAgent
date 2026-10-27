"""Agent 核心：实现 ReAct（Reasoning + Acting）循环。

一次 run() 的完整链路::

    user 输入
      └─> [循环] 拼装 prompt ─> 调模型 ─> 解析输出
                                  ├─ Final Answer ─> 结束
                                  └─ Action ─> 校验参数 ─> 执行工具
                                               └─> Observation 回灌上下文 ─> 下一轮

三个工程上必须处理好的边界：
    1. 模型输出不合协议 —— 不崩溃，把纠错提示作为 Observation 回灌，让它自我修正。
    2. 模型原地打转 —— 同一动作+同样参数重复多次则判定死循环，立即收尾。
    3. 步数耗尽 —— 追加一条「禁止再调用工具」的指令，强制它基于已有观察作答。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .llm import LLM, Message
from .memory import ConversationMemory
from .parser import parse
from .prompts import FINALIZE_PROMPT, build_system_prompt
from .tools import build_tool_prompt, execute_tool
from .tracer import Tracer

_OBSERVATION_TEMPLATE = "Observation: {content}"
_FORMAT_HINT = (
    "你的上一次输出不符合协议：既没有可解析的 Action，也没有 Final Answer。"
    "请严格按下面的格式重新输出，不要添加任何解释性文字：\n"
    "Thought: ...\nAction: 工具名\nAction Input: {{\"参数名\": \"参数值\"}}\n"
    "如果已经可以回答，请用 Final Answer: ..."
)

# 事件回调签名：kind + 结构化数据，CLI 用它做实时流式打印
EventHandler = Callable[[str, dict[str, Any]], None]


@dataclass(slots=True)
class Step:
    """循环中的一步。"""

    index: int
    thought: str
    action: str | None = None
    action_input: Any = None
    observation: str | None = None
    ok: bool = True
    elapsed_ms: float = 0.0
    is_final: bool = False
    executed: bool = False  # 工具是否真的被执行过（被死循环保护拦下的不算）

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
            "ok": self.ok,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "is_final": self.is_final,
            "executed": self.executed,
        }


@dataclass
class AgentResult:
    """一次运行的完整结果。"""

    answer: str
    steps: list[Step] = field(default_factory=list)
    stop_reason: str = "final"  # final | max_steps | loop_detected | error
    elapsed_ms: float = 0.0
    trace_path: str | None = None

    @property
    def tool_calls(self) -> int:
        """实际执行过的工具调用次数（不含被拦下的重复动作）。"""
        return sum(1 for s in self.steps if s.executed)

    def render(self, show_thought: bool = True) -> str:
        """把执行轨迹渲染成人类可读的文本。"""
        lines: list[str] = []
        for step in self.steps:
            lines.append(f"[步骤 {step.index + 1}]")
            if show_thought and step.thought:
                lines.append(f"  思考: {_one_line(step.thought, 200)}")
            if step.action:
                params = json.dumps(step.action_input, ensure_ascii=False, default=str)
                lines.append(f"  动作: {step.action}({_one_line(params, 160)})")
                status = "" if step.ok else "  [失败]"
                lines.append(f"  观察: {_one_line(step.observation or '', 400)}{status}")
                lines.append(f"  耗时: {step.elapsed_ms:.1f}ms")
            lines.append("")

        lines.append(f"停止原因: {self.stop_reason} | 工具调用 {self.tool_calls} 次 | 总耗时 {self.elapsed_ms:.0f}ms")
        lines.append(f"最终回答: {self.answer}")
        if self.trace_path:
            lines.append(f"执行轨迹: {self.trace_path}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "stop_reason": self.stop_reason,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "tool_calls": self.tool_calls,
            "steps": [s.to_dict() for s in self.steps],
            "trace_path": self.trace_path,
        }


class Agent:
    """ReAct 智能体。

    参数:
        llm:         任意实现 ``chat(messages) -> str`` 的模型对象
        memory:      对话记忆，默认 8 轮 / 12000 字符预算
        max_steps:   单次任务最大循环步数
        max_repeats: 同一动作重复多少次判定为死循环
        tracer:      轨迹记录器，默认关闭
        on_event:    事件回调，用于实时打印
    """

    def __init__(
        self,
        llm: LLM,
        memory: ConversationMemory | None = None,
        *,
        max_steps: int = 8,
        max_repeats: int = 2,
        tracer: Tracer | None = None,
        on_event: EventHandler | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps 必须 >= 1")
        self.llm = llm
        self.memory = memory or ConversationMemory()
        self.max_steps = max_steps
        self.max_repeats = max_repeats
        self.tracer = tracer or Tracer(enabled=False)
        self.on_event = on_event
        self.system_prompt = build_system_prompt(build_tool_prompt())

    # ------------------------------------------------------------------ 主流程
    def run(self, user_input: str) -> AgentResult:
        started = time.perf_counter()
        self.memory.add_user(user_input)
        self.tracer.event(
            "run_start",
            input=user_input,
            model=getattr(self.llm, "model_name", "unknown"),
            max_steps=self.max_steps,
        )
        self._emit("run_start", {"input": user_input})

        steps: list[Step] = []
        answer = ""
        stop_reason = "max_steps"
        signature_counts: dict[str, int] = {}

        for index in range(self.max_steps):
            raw, llm_ms, llm_error = self._call_llm(index)
            if raw is None:  # 模型调用失败
                stop_reason = "error"
                answer = f"模型调用失败：{llm_error}"
                break

            parsed = parse(raw)
            self.memory.add_assistant(raw)

            # --- 情况 1：给出最终答案
            if parsed.kind == "final":
                step = Step(index=index, thought=parsed.thought, is_final=True, elapsed_ms=llm_ms)
                steps.append(step)
                answer = (parsed.final_answer or "").strip()
                stop_reason = "final"
                self._record_step(step)
                break

            # --- 情况 2：没解析出任何动作，把格式要求回灌让它重试
            if parsed.kind == "empty":
                step = Step(
                    index=index,
                    thought=parsed.thought,
                    ok=False,
                    observation="输出格式不符合协议",
                    elapsed_ms=llm_ms,
                )
                steps.append(step)
                self._record_step(step)
                self.memory.add_user(_OBSERVATION_TEMPLATE.format(content=_FORMAT_HINT))
                continue

            # --- 情况 3：调用工具
            signature = self._signature(parsed.action, parsed.action_input)
            signature_counts[signature] = signature_counts.get(signature, 0) + 1
            if signature_counts[signature] > self.max_repeats:
                step = Step(
                    index=index,
                    thought=parsed.thought,
                    action=parsed.action,
                    action_input=parsed.action_input,
                    ok=False,
                    observation="检测到重复动作，已中止循环",
                    elapsed_ms=llm_ms,
                )
                steps.append(step)
                self._record_step(step)
                stop_reason = "loop_detected"
                break

            result = execute_tool(parsed.action or "", parsed.action_input)
            step = Step(
                index=index,
                thought=parsed.thought,
                action=parsed.action,
                action_input=parsed.action_input,
                observation=result.content,
                ok=result.ok,
                elapsed_ms=result.elapsed_ms,
                executed=True,
            )
            steps.append(step)
            self._record_step(step)
            self.tracer.event(
                "tool_result",
                step=index,
                tool=result.tool,
                ok=result.ok,
                elapsed_ms=round(result.elapsed_ms, 2),
                output=_one_line(result.content, 500),
            )

            self.memory.add_user(_OBSERVATION_TEMPLATE.format(content=result.content))

        # --- 收尾：没拿到答案就强制让它总结
        if not answer:
            answer = self._force_finalize(stop_reason)

        elapsed_ms = (time.perf_counter() - started) * 1000
        result = AgentResult(
            answer=answer,
            steps=steps,
            stop_reason=stop_reason,
            elapsed_ms=elapsed_ms,
            trace_path=self.tracer.path_str,
        )
        self.tracer.event(
            "run_end",
            stop_reason=stop_reason,
            tool_calls=result.tool_calls,
            elapsed_ms=round(elapsed_ms, 2),
            answer=_one_line(answer, 500),
        )
        self._emit("run_end", {"result": result})
        return result

    # ------------------------------------------------------------------ 内部方法
    def _call_llm(self, index: int) -> tuple[str | None, float, str | None]:
        """调用模型。返回 (输出, 耗时, 错误信息)，失败时输出为 None。"""
        messages: list[Message] = [Message("system", self.system_prompt)]
        messages.extend(self.memory.messages())

        started = time.perf_counter()
        try:
            raw = self.llm.chat(messages, stop=["Observation:", "观察:"])
        except Exception as exc:  # noqa: BLE001 - 模型层异常统一降级
            elapsed_ms = (time.perf_counter() - started) * 1000
            message = _one_line(str(exc), 200) or type(exc).__name__
            self.tracer.event("error", where="llm", step=index, message=message)
            self._emit("error", {"where": "llm", "message": message})
            return None, elapsed_ms, message

        elapsed_ms = (time.perf_counter() - started) * 1000
        self.tracer.event(
            "llm_call",
            step=index,
            elapsed_ms=round(elapsed_ms, 2),
            output=raw,
        )
        return raw, elapsed_ms, None

    def _force_finalize(self, stop_reason: str) -> str:
        """步数用尽或检测到死循环时，强制模型基于已有观察给出结论。"""
        messages: list[Message] = [Message("system", self.system_prompt)]
        messages.extend(self.memory.messages())
        messages.append(Message("user", FINALIZE_PROMPT))

        try:
            raw = self.llm.chat(messages, stop=["Observation:"])
        except Exception as exc:  # noqa: BLE001
            self.tracer.event("error", where="finalize", message=str(exc))
            return f"任务未在 {self.max_steps} 步内完成（{stop_reason}），且总结阶段调用模型失败。"

        parsed = parse(raw)
        self.tracer.event("finalize", stop_reason=stop_reason, output=raw)
        answer = (parsed.final_answer or raw).strip()
        return answer or f"任务未在 {self.max_steps} 步内完成（{stop_reason}）。"

    def _record_step(self, step: Step) -> None:
        self.tracer.event("step", **step.to_dict())
        self._emit("step", {"step": step})

    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        if self.on_event:
            self.on_event(kind, data)

    @staticmethod
    def _signature(action: str | None, action_input: Any) -> str:
        try:
            payload = json.dumps(action_input, ensure_ascii=False, sort_keys=True, default=str)
        except (TypeError, ValueError):
            payload = repr(action_input)
        return f"{action}|{payload}"

    # ------------------------------------------------------------------ 会话控制
    def reset(self) -> None:
        self.memory.reset()


def _one_line(text: str, limit: int) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
