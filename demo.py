"""一键演示：用离线 Mock 模型跑通 4 个典型 Agent 任务。

运行::

    python demo.py

它不需要任何 API Key，也不需要联网——目的就是让别人 clone 下来
第一时间看到「Agent 到底在做什么」。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from miniagent import Agent, MockLLM, Tracer

CASES = [
    "帮我算一下 (12+8)*3/5 等于多少",
    "现在几点了",
    "记个待办：复习 ReAct 循环的实现",
    "看看工作区里有哪些文件",
    "算一下 2**10 是多少，然后记个待办：验证计算器工具",
]


def _isolate_workspace() -> None:
    """演示使用临时目录，避免把 demo 数据写进真实工作区。"""
    tmp = Path(tempfile.mkdtemp(prefix="miniagent-demo-"))
    os.environ["MINIAGENT_TODO_DB"] = str(tmp / "todo.db")
    os.environ["MINIAGENT_SANDBOX"] = str(tmp / "workspace")
    (tmp / "workspace").mkdir(parents=True, exist_ok=True)
    (tmp / "workspace" / "hello.md").write_text("# 这是工作区里的示例文件\n", encoding="utf-8")


def main() -> None:
    _isolate_workspace()
    print("=" * 62)
    print("MiniAgent 演示（离线 Mock 模式，无需 API Key）")
    print("=" * 62)

    agent = Agent(
        llm=MockLLM(),
        max_steps=6,
        tracer=Tracer(enabled=False),
    )

    for i, case in enumerate(CASES, 1):
        print(f"\n{'#' * 62}\n# 场景 {i}：{case}\n{'#' * 62}")
        result = agent.run(case)
        print(result.render())
        print()

    print("=" * 62)
    print("演示结束。配置 OPENAI_API_KEY 后可切换真实模型：")
    print("  python -m miniagent \"你的问题\" --provider openai --model gpt-4o-mini")


if __name__ == "__main__":
    main()
