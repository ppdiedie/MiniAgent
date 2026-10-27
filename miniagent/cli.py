"""命令行入口。

用法::

    python -m miniagent "帮我算一下 (12+8)*3/5"      # 单次任务
    python -m miniagent -i                            # 交互模式，多轮记忆
    python -m miniagent --list-tools                   # 查看已注册工具
    python -m miniagent "..." --json                   # 机器可读输出
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from .agent import Agent, AgentResult
from .llm import LLMError, OpenAICompatibleLLM
from .memory import ConversationMemory
from .mock_llm import MockLLM
from .tools import list_tools
from .tracer import Tracer

_DIVIDER = "-" * 62


# --------------------------------------------------------------------------- 参数
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="miniagent",
        description="MiniAgent —— 轻量级 ReAct 智能体框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("query", nargs="*", help="要交给 Agent 的任务")
    p.add_argument("-i", "--interactive", action="store_true", help="进入交互模式，保留多轮记忆")
    p.add_argument("--provider", choices=["auto", "mock", "openai"], default="auto",
                   help="模型来源：mock 为离线规则引擎，openai 为兼容接口（默认按是否有 API Key 自动选择）")
    p.add_argument("--model", default=None, help="模型名称，如 gpt-4o-mini / deepseek-chat")
    p.add_argument("--base-url", default=None, help="OpenAI 兼容接口地址")
    p.add_argument("--max-steps", type=int, default=8, help="单次任务最大循环步数")
    p.add_argument("--json", action="store_true", help="以 JSON 格式输出结果")
    p.add_argument("--no-trace", action="store_true", help="不写执行轨迹文件")
    p.add_argument("--list-tools", action="store_true", help="列出所有已注册工具后退出")
    p.add_argument("-q", "--quiet", action="store_true", help="只输出最终答案")
    return p


# --------------------------------------------------------------------------- 装配
def build_agent(args: argparse.Namespace) -> Agent:
    provider = args.provider
    if provider == "auto":
        provider = "openai" if os.getenv("OPENAI_API_KEY") else "mock"

    if provider == "mock":
        llm: Any = MockLLM()
        if not args.quiet:
            print("[模式] 离线 Mock 引擎（未配置 API Key，仅演示预设任务）\n")
    else:
        llm = OpenAICompatibleLLM(model=args.model, base_url=args.base_url)

    tracer = Tracer(enabled=not args.no_trace)
    return Agent(
        llm=llm,
        memory=ConversationMemory(),
        max_steps=args.max_steps,
        tracer=tracer,
        on_event=None if args.quiet or args.json else _print_event,
    )


# --------------------------------------------------------------------------- 输出
def _print_event(kind: str, data: dict[str, Any]) -> None:
    if kind == "step":
        step = data["step"]
        if step.is_final:
            return
        if step.thought:
            print(f"  思考: {_one_line(step.thought, 180)}")
        if step.action:
            params = json.dumps(step.action_input, ensure_ascii=False, default=str)
            print(f"  动作: {step.action} {_one_line(params, 140)}")
            flag = "  [失败]" if not step.ok else ""
            print(f"  观察: {_one_line(step.observation or '', 300)}{flag}  ({step.elapsed_ms:.1f}ms)")
            print()
    elif kind == "error":
        print(f"  [错误] {data.get('message')}", file=sys.stderr)


def _one_line(text: str, limit: int) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def render_result(result: AgentResult, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return

    print(_DIVIDER)
    print(result.answer)
    print(_DIVIDER)
    print(
        f"停止原因={result.stop_reason} | 工具调用={result.tool_calls} 次 | 耗时={result.elapsed_ms:.0f}ms"
        + (f" | 轨迹={result.trace_path}" if result.trace_path else "")
    )


# --------------------------------------------------------------------------- 主流程
def run_once(agent: Agent, query: str, args: argparse.Namespace) -> int:
    try:
        result = agent.run(query)
    except LLMError as exc:
        print(f"模型调用失败：{exc}", file=sys.stderr)
        return 2
    render_result(result, args.json)
    return 0


def run_interactive(agent: Agent, args: argparse.Namespace) -> int:
    print("MiniAgent 交互模式。输入 exit / quit 退出，输入 reset 清空上下文。")
    print(_DIVIDER)
    while True:
        try:
            query = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return 0

        if not query:
            continue
        if query.lower() in {"exit", "quit", ":q"}:
            print("已退出。")
            return 0
        if query.lower() == "reset":
            agent.reset()
            print("上下文已清空。")
            continue

        try:
            result = agent.run(query)
        except LLMError as exc:
            print(f"模型调用失败：{exc}", file=sys.stderr)
            continue

        print(_DIVIDER)
        print(result.answer)
        print(_DIVIDER)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_tools:
        for spec in list_tools():
            required = spec.parameters.get("required", [])
            print(f"{spec.name:<12} {spec.description}")
            if required:
                print(f"{'':<12} 必填参数: {', '.join(required)}")
        return 0

    agent = build_agent(args)

    if args.interactive:
        return run_interactive(agent, args)

    query = " ".join(args.query).strip()
    if not query:
        build_parser().print_help()
        return 1
    return run_once(agent, query, args)


if __name__ == "__main__":
    raise SystemExit(main())
