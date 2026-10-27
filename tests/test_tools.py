"""工具层测试：安全性、参数校验、异常隔离。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from miniagent.tools import execute_tool, get_tool, list_tools
from miniagent.tools.calculator import calculator
from miniagent.tools.files import list_dir, read_file, write_file
from miniagent.tools.todo import todo_add, todo_done, todo_list


class SandboxMixin(unittest.TestCase):
    """把文件沙箱和待办数据库都重定向到临时目录。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._saved = {
            "MINIAGENT_SANDBOX": os.environ.get("MINIAGENT_SANDBOX"),
            "MINIAGENT_TODO_DB": os.environ.get("MINIAGENT_TODO_DB"),
        }
        root = Path(self._tmp.name)
        os.environ["MINIAGENT_SANDBOX"] = str(root / "workspace")
        os.environ["MINIAGENT_TODO_DB"] = str(root / "todo.db")

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._tmp.cleanup()


# --------------------------------------------------------------------------- 计算器
class TestCalculator(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(calculator("(12+8)*3/5"), "12")

    def test_power(self):
        self.assertEqual(calculator("2**10"), "1024")

    def test_float_formatting(self):
        self.assertEqual(calculator("1/3"), "0.333333")

    def test_math_functions(self):
        self.assertEqual(calculator("sqrt(16)"), "4")
        self.assertEqual(calculator("max(1, 5, 3)"), "5")

    def test_constants(self):
        self.assertTrue(calculator("pi").startswith("3.14159"))

    def test_division_by_zero(self):
        with self.assertRaises(ValueError):
            calculator("1/0")

    def test_reject_code_injection(self):
        """eval 会把下面这些直接执行掉——AST 白名单必须拦住。"""
        attacks = [
            "__import__('os').system('echo hacked')",
            "open('x.txt','w')",
            "(lambda: 1)()",
            "[].__class__",
            "1 if True else 2",
        ]
        for expr in attacks:
            with self.subTest(expr=expr):
                with self.assertRaises(ValueError):
                    calculator(expr)

    def test_reject_power_bomb(self):
        # 9**9**9 会让纯 eval 实现直接卡死
        with self.assertRaises(ValueError):
            calculator("9**9**9")

    def test_reject_too_long(self):
        with self.assertRaises(ValueError):
            calculator("1+" * 400 + "1")

    def test_reject_empty(self):
        with self.assertRaises(ValueError):
            calculator("   ")


# --------------------------------------------------------------------------- 文件
class TestFiles(SandboxMixin):
    def test_write_then_read(self):
        write_file("notes/a.md", "# 标题")
        self.assertEqual(read_file("notes/a.md"), "# 标题")

    def test_list_dir(self):
        write_file("x.txt", "hi")
        out = list_dir(".")
        self.assertIn("x.txt", out)

    def test_list_empty_dir(self):
        self.assertIn("空目录", list_dir("."))

    def test_path_traversal_blocked(self):
        for bad in ["../secret.txt", "../../etc/passwd", "a/../../outside.txt"]:
            with self.subTest(path=bad):
                with self.assertRaises(ValueError):
                    read_file(bad)

    def test_absolute_path_blocked(self):
        for bad in ["C:/Windows/system.ini", "/etc/hosts"]:
            with self.subTest(path=bad):
                with self.assertRaises(ValueError):
                    read_file(bad)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            read_file("nope.txt")

    def test_non_text_suffix(self):
        write_file("image.bin", "x")
        with self.assertRaises(ValueError):
            read_file("image.bin")


# --------------------------------------------------------------------------- 待办
class TestTodo(SandboxMixin):
    def test_add_and_list(self):
        todo_add("写周报")
        out = todo_list()
        self.assertIn("写周报", out)
        self.assertIn("[ ]", out)

    def test_mark_done(self):
        todo_add("刷题")
        todo_done(1)
        self.assertIn("[x] #1", todo_list())

    def test_done_twice(self):
        todo_add("刷题")
        todo_done(1)
        self.assertIn("已经是完成状态", todo_done(1))

    def test_done_missing(self):
        with self.assertRaises(ValueError):
            todo_done(999)

    def test_list_empty(self):
        self.assertIn("没有待办", todo_list())

    def test_sql_injection_is_data(self):
        """参数化查询：恶意字符串只会被当成普通文本存下来。"""
        payload = "'); DROP TABLE todo; --"
        todo_add(payload)
        self.assertIn("DROP TABLE", todo_list())  # 表还在，数据也在

    def test_empty_title(self):
        with self.assertRaises(ValueError):
            todo_add("   ")


# --------------------------------------------------------------------------- 校验层
class TestExecuteTool(unittest.TestCase):
    def test_unknown_tool(self):
        result = execute_tool("no_such_tool", {})
        self.assertFalse(result.ok)
        self.assertIn("不存在", result.content)

    def test_missing_required(self):
        result = execute_tool("calculator", {})
        self.assertFalse(result.ok)
        self.assertIn("缺少必填参数", result.content)

    def test_unknown_argument(self):
        result = execute_tool("calculator", {"expression": "1+1", "evil": 1})
        self.assertFalse(result.ok)
        self.assertIn("未知参数", result.content)

    def test_type_mismatch(self):
        result = execute_tool("calculator", {"expression": 123})
        self.assertFalse(result.ok)
        self.assertIn("期望 string", result.content)

    def test_bare_value_binding(self):
        """模型只给裸值时，自动绑定到唯一参数。"""
        result = execute_tool("calculator", {"__value__": "7*6"})
        self.assertTrue(result.ok)
        self.assertEqual(result.content, "42")

    def test_tool_exception_isolated(self):
        result = execute_tool("calculator", {"expression": "1/0"})
        self.assertFalse(result.ok)
        self.assertIn("除数不能为 0", result.content)

    def test_success_payload(self):
        result = execute_tool("calculator", {"expression": "1+1"})
        self.assertTrue(result.ok)
        self.assertEqual(result.content, "2")
        self.assertGreaterEqual(result.elapsed_ms, 0)

    def test_schema_export(self):
        spec = get_tool("calculator")
        schema = spec.to_openai_schema()
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "calculator")
        self.assertIn("expression", schema["function"]["parameters"]["properties"])

    def test_all_builtin_registered(self):
        names = {s.name for s in list_tools()}
        expected = {"calculator", "now", "read_file", "write_file", "list_dir",
                    "todo_add", "todo_done", "todo_list"}
        self.assertEqual(names, expected)


if __name__ == "__main__":
    unittest.main()
