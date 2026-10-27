"""Agent 循环测试。

用「脚本化模型」替代真实 LLM：每个用例精确控制模型每一步输出什么，
从而把循环调度、死循环检测、异常恢复这些逻辑测到确定性的边界。
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

from miniagent.agent import Agent
from miniagent.llm import Message
from miniagent.memory import ConversationMemory


class ScriptedLLM:
    """按预设脚本逐条返回，超出脚本长度后重复最后一条。"""

    model_name = "scripted"

    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.history: list[list[Message]] = []

    def chat(self, messages: Sequence[Message], stop: Sequence[str] | None = None) -> str:
        self.history.append(list(messages))
        index = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[index]


class ExplodingLLM:
    model_name = "exploding"

    def chat(self, messages: Sequence[Message], stop: Sequence[str] | None = None) -> str:
        raise RuntimeError("模拟网络中断")


def _action(expr: str) -> str:
    return f'Thought: 计算 {expr}\nAction: calculator\nAction Input: {{"expression": "{expr}"}}'


FINAL = "Thought: 已有答案\nFinal Answer: 完成"


# --------------------------------------------------------------------------- 正常路径
class TestHappyPath(unittest.TestCase):
    def test_single_tool_call(self):
        agent = Agent(ScriptedLLM(_action("6*7"), FINAL))
        result = agent.run("6*7 是多少")

        self.assertEqual(result.stop_reason, "final")
        self.assertEqual(result.answer, "完成")
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual(result.steps[0].observation, "42")
        self.assertTrue(result.steps[0].ok)

    def test_multi_step(self):
        agent = Agent(ScriptedLLM(_action("1+1"), _action("2+2"), FINAL))
        result = agent.run("算两次")

        self.assertEqual(result.tool_calls, 2)
        self.assertEqual([s.observation for s in result.steps[:2]], ["2", "4"])

    def test_observation_fed_back_to_model(self):
        llm = ScriptedLLM(_action("3*3"), FINAL)
        agent = Agent(llm)
        agent.run("3*3")

        # 第二次调用模型时，上下文里必须包含 Observation: 9
        second_call = llm.history[1]
        self.assertTrue(any(m.content.startswith("Observation: 9") for m in second_call))

    def test_direct_answer_without_tool(self):
        agent = Agent(ScriptedLLM(FINAL))
        result = agent.run("你好")
        self.assertEqual(result.stop_reason, "final")
        self.assertEqual(result.tool_calls, 0)

    def test_result_serializable(self):
        agent = Agent(ScriptedLLM(_action("2*3"), FINAL))
        data = agent.run("2*3").to_dict()
        self.assertEqual(data["tool_calls"], 1)
        self.assertEqual(data["steps"][0]["observation"], "6")
        self.assertIn("stop_reason", data)


# --------------------------------------------------------------------------- 异常与边界
class TestRobustness(unittest.TestCase):
    def test_unknown_tool_does_not_crash(self):
        bad = 'Thought: 试试\nAction: send_email\nAction Input: {"to": "x"}'
        agent = Agent(ScriptedLLM(bad, FINAL))
        result = agent.run("发邮件")

        self.assertEqual(result.stop_reason, "final")
        self.assertFalse(result.steps[0].ok)
        self.assertIn("不存在", result.steps[0].observation)

    def test_bad_format_gets_retried(self):
        # 第一步只有 Thought，没有 Action/Final → 应回灌格式提示后继续
        agent = Agent(ScriptedLLM("Thought: 我在想", FINAL))
        result = agent.run("随便问点")

        self.assertEqual(result.stop_reason, "final")
        self.assertFalse(result.steps[0].ok)
        self.assertEqual(len(result.steps), 2)

    def test_loop_detected(self):
        same = _action("6*7")
        agent = Agent(ScriptedLLM(same, same, same, same), max_steps=6, max_repeats=2)
        result = agent.run("一直算 6*7")

        self.assertEqual(result.stop_reason, "loop_detected")
        self.assertEqual(result.tool_calls, 2)  # 第 3 次相同动作被拦下

    def test_max_steps_force_finalize(self):
        llm = ScriptedLLM(_action("1+1"), _action("2+2"), _action("3+3"), "Thought: 收尾\nFinal Answer: 已尽力")
        agent = Agent(llm, max_steps=3)
        result = agent.run("无限任务")

        self.assertEqual(result.stop_reason, "max_steps")
        self.assertEqual(result.tool_calls, 3)
        self.assertEqual(result.answer, "已尽力")

    def test_llm_failure_returns_error_reason(self):
        agent = Agent(ExplodingLLM())
        result = agent.run("hi")

        self.assertEqual(result.stop_reason, "error")
        self.assertIn("模型调用失败", result.answer)

    def test_tool_exception_is_observation(self):
        bad = 'Action: calculator\nAction Input: {"expression": "1/0"}'
        agent = Agent(ScriptedLLM(bad, FINAL))
        result = agent.run("1/0")

        self.assertFalse(result.steps[0].ok)
        self.assertIn("除数不能为 0", result.steps[0].observation)

    def test_max_steps_validation(self):
        with self.assertRaises(ValueError):
            Agent(ScriptedLLM(FINAL), max_steps=0)


# --------------------------------------------------------------------------- 记忆
class TestMemory(unittest.TestCase):
    def test_turn_trimming(self):
        memory = ConversationMemory(max_turns=4)
        for i in range(10):
            memory.add_user(f"问题{i}")
            memory.add_assistant(f"回答{i}")

        messages = memory.messages()
        self.assertEqual(len(messages), 5)  # 1 条省略提示 + 4 条
        self.assertTrue(messages[0].content.startswith("[上下文已裁剪"))
        self.assertEqual(messages[0].role, "user")

    def test_char_budget(self):
        memory = ConversationMemory(max_turns=100, max_chars=120)
        for i in range(10):
            memory.add_user("x" * 40)
            memory.add_assistant("y" * 40)

        messages = memory.messages()
        total = sum(len(m.content) for m in messages)
        self.assertLess(total, 400)
        self.assertEqual(messages[0].role, "user")

    def test_reset(self):
        agent = Agent(ScriptedLLM(FINAL))
        agent.run("第一轮")
        self.assertGreater(len(agent.memory), 0)
        agent.reset()
        self.assertEqual(len(agent.memory), 0)

    def test_multi_turn_keeps_history(self):
        agent = Agent(ScriptedLLM(FINAL))
        agent.run("第一轮")
        agent.run("第二轮")
        self.assertEqual(len(agent.memory.messages()) >= 4, True)


# --------------------------------------------------------------------------- 轨迹
class TestTracer(unittest.TestCase):
    def test_trace_file_written(self):
        from miniagent.tracer import Tracer

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            tracer = Tracer(path=path)
            agent = Agent(ScriptedLLM(_action("5*5"), FINAL), tracer=tracer)
            result = agent.run("5*5")

            self.assertEqual(result.trace_path, str(path))
            content = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertGreaterEqual(len(content), 4)  # run_start / step / llm_call / run_end

            import json

            events = [json.loads(line)["event"] for line in content]
            self.assertIn("run_start", events)
            self.assertIn("run_end", events)
            self.assertEqual(tracer.summary()["step"], 2)


if __name__ == "__main__":
    unittest.main()
