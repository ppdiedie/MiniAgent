"""解析器测试。

重点覆盖模型输出的各种「不像话」写法——这类边界是 Agent 线上故障的高发区。
"""

from __future__ import annotations

import unittest

from miniagent.parser import parse


class TestFinalAnswer(unittest.TestCase):
    def test_plain_final(self):
        step = parse("Thought: 够了\nFinal Answer: 结果是 12")
        self.assertEqual(step.kind, "final")
        self.assertEqual(step.final_answer, "结果是 12")
        self.assertIn("够了", step.thought)

    def test_final_multiline(self):
        text = "Thought: ok\nFinal Answer: 第一行\n第二行"
        step = parse(text)
        self.assertEqual(step.final_answer, "第一行\n第二行")

    def test_chinese_colon(self):
        step = parse("思考：已经完成\nFinal Answer：完成")
        self.assertEqual(step.kind, "final")
        self.assertEqual(step.final_answer, "完成")

    def test_final_priority_over_action(self):
        # 模型同时吐了 Action 和 Final Answer，应以 Final 为准
        text = "Thought: t\nAction: calculator\nAction Input: {\"expression\": \"1+1\"}\nFinal Answer: 直接回答"
        step = parse(text)
        self.assertEqual(step.kind, "final")
        self.assertEqual(step.final_answer, "直接回答")

    def test_observation_echo_stripped(self):
        # 模型自造 Observation，必须在解析阶段切掉
        text = "Thought: t\nFinal Answer: 答案 42\nObservation: 这是模型编的"
        step = parse(text)
        self.assertEqual(step.final_answer, "答案 42")


class TestAction(unittest.TestCase):
    def test_basic_action(self):
        text = 'Thought: 要算一下\nAction: calculator\nAction Input: {"expression": "(12+8)*3/5"}'
        step = parse(text)
        self.assertEqual(step.kind, "action")
        self.assertEqual(step.action, "calculator")
        self.assertEqual(step.action_input, {"expression": "(12+8)*3/5"})

    def test_markdown_fence(self):
        text = 'Thought: t\n```\nAction: now\nAction Input: {}\n```'
        step = parse(text)
        self.assertEqual(step.action, "now")

    def test_fence_with_language_tag(self):
        text = "```json\nAction: todo_list\nAction Input: {\"only_pending\": true}\n```"
        step = parse(text)
        self.assertEqual(step.action, "todo_list")
        self.assertEqual(step.action_input, {"only_pending": True})

    def test_single_quotes(self):
        text = "Action: calculator\nAction Input: {'expression': '1+1'}"
        step = parse(text)
        self.assertEqual(step.action_input, {"expression": "1+1"})

    def test_trailing_comma(self):
        text = 'Action: calculator\nAction Input: {"expression": "1+1",}'
        step = parse(text)
        self.assertEqual(step.action_input, {"expression": "1+1"})

    def test_inline_call_syntax(self):
        text = 'Action: calculator({"expression": "2**8"})'
        step = parse(text)
        self.assertEqual(step.action, "calculator")
        self.assertEqual(step.action_input, {"expression": "2**8"})

    def test_bare_value_input(self):
        text = "Action: calculator\nAction Input: (1+2)*3"
        step = parse(text)
        self.assertEqual(step.action, "calculator")
        # 裸值原样保留，由运行时绑定到唯一参数
        self.assertEqual(step.action_input, "(1+2)*3")

    def test_action_name_trailing_punctuation(self):
        step = parse("Action: calculator。\nAction Input: {\"expression\": \"1\"}")
        self.assertEqual(step.action, "calculator")

    def test_action_without_input(self):
        step = parse("Action: todo_list")
        self.assertEqual(step.action, "todo_list")
        self.assertEqual(step.action_input, {})


class TestFallback(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(parse("").kind, "empty")
        self.assertEqual(parse("   \n  ").kind, "empty")

    def test_plain_text_treated_as_final(self):
        # 弱模型经常直接给答案，不应判为错误
        step = parse("北京今天晴，气温 20 度。")
        self.assertEqual(step.kind, "final")
        self.assertIn("北京", step.final_answer)

    def test_thought_only_is_empty(self):
        step = parse("Thought: 我在思考")
        self.assertEqual(step.kind, "empty")

    def test_never_raises(self):
        for weird in ["{{{", "Action:", "Final Answer:", "\x00", "```\n```"]:
            parse(weird)  # 不抛异常即通过


if __name__ == "__main__":
    unittest.main()
