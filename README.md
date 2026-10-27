# MiniAgent

[![Tests](https://github.com/ppdiedie/MiniAgent/actions/workflows/test.yml/badge.svg)](https://github.com/ppdiedie/MiniAgent/actions/workflows/test.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Dependencies](https://img.shields.io/badge/dependencies-0-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

> 从零实现的轻量级 **ReAct 智能体框架**。不依赖 LangChain / LlamaIndex，只用 Python 标准库，
> clone 下来一条命令即可运行，无需 API Key。

MiniAgent 实现了一个完整的智能体运行闭环：**思考 → 调用工具 → 观察结果 → 继续思考**，
直到得出答案。它的目标不是做一个功能大而全的框架，而是把 Agent 的核心机制
（循环调度、工具协议、上下文管理、异常恢复）用可读、可测的代码完整表达出来。

```
用户: 算一下 2**10 是多少，然后记个待办：验证计算器工具

[步骤 1] 思考: 用户给出了数学表达式，我用 calculator 精确求值，避免心算出错。
         动作: calculator({"expression": "2**10"})
         观察: 1024                                        (0.0ms)

[步骤 2] 思考: 把这件要做的事登记到待办清单里。
         动作: todo_add({"title": "验证计算器工具"})
         观察: 已添加待办 #2：验证计算器工具                  (7.0ms)

[步骤 3] 思考: 我已经拿到需要的信息。
最终回答: 计算结果：1024 / 待办已记录：验证计算器工具
停止原因: final | 工具调用 2 次 | 总耗时 8ms
```

---

## 快速开始

无需安装任何第三方包，Python 3.10+ 即可：

```bash
git clone https://github.com/ppdiedie/MiniAgent.git && cd miniagent

# 1. 看演示（离线，跑通 5 个典型任务）
python demo.py

# 2. 单次执行
python -m miniagent "帮我算一下 (12+8)*3/5"

# 3. 交互模式（多轮记忆）
python -m miniagent -i

# 4. 查看已注册的工具
python -m miniagent --list-tools

# 5. 运行测试
python -m unittest discover -s tests -t . -v
```

### 接入真实模型

框架对模型的要求只有一个方法：`chat(messages) -> str`。内置的 `OpenAICompatibleLLM`
可直接对接 OpenAI / DeepSeek / 阿里百炼 / 智谱 / 本地 vLLM 等任意兼容接口：

```bash
# OpenAI
export OPENAI_API_KEY=sk-xxx
python -m miniagent "你的问题" --provider openai --model gpt-4o-mini

# DeepSeek
export OPENAI_API_KEY=sk-xxx
export OPENAI_BASE_URL=https://api.deepseek.com/v1
python -m miniagent "你的问题" --provider openai --model deepseek-chat

# 本地 Ollama（无需 Key）
export OPENAI_API_KEY=ollama
export OPENAI_BASE_URL=http://localhost:11434/v1
python -m miniagent "你的问题" --provider openai --model qwen2.5:7b
```

> 未配置 `OPENAI_API_KEY` 时，程序自动降级到离线 Mock 引擎，不会报错中断。

---

## 架构

```
                        ┌──────────────┐
        用户输入 ───────▶│   Memory     │  轮数裁剪 + 字符预算裁剪
                        └──────┬───────┘
                               │ messages
                               ▼
                        ┌──────────────┐
                        │  LLM 适配层   │  OpenAI 兼容接口 / 离线 Mock
                        └──────┬───────┘
                               │ 纯文本
                               ▼
                        ┌──────────────┐
                        │    Parser    │  容错解析（fence/单引号/裸值/全角冒号）
                        └──────┬───────┘
                    ┌──────────┴──────────┐
              Final Answer             Action
                    │                    │
                    ▼                    ▼
                返回用户          ┌──────────────┐
                                  │  工具注册表   │
                                  └──────┬───────┘
                                         │ 参数校验 → 沙箱执行
                                         ▼
                                   Observation
                                         │
                                         └──▶ 回灌 Memory，进入下一轮
```

### 目录结构

```
miniagent/
├── miniagent/
│   ├── agent.py        # ReAct 循环调度、死循环检测、强制收尾        (329 行)
│   ├── parser.py       # 模型输出容错解析                           (189 行)
│   ├── llm.py          # 模型适配层（urllib 直连 + 重试退避）        (144 行)
│   ├── memory.py       # 双层上下文裁剪                             (88 行)
│   ├── mock_llm.py     # 离线规则引擎，使测试不依赖网络与费用
│   ├── prompts.py      # 提示词模板
│   ├── tracer.py       # 结构化执行轨迹（JSONL）
│   ├── cli.py          # 命令行入口
│   └── tools/
│       ├── base.py     # 注册表 / Schema 生成 / 参数校验 / 异常隔离
│       ├── calculator.py  # AST 白名单安全求值
│       ├── files.py       # 工作区沙箱读写（防路径穿越）
│       ├── todo.py        # SQLite 持久化（参数化查询）
│       └── clock.py
├── tests/              # 68 个单元测试，550 行
├── demo.py             # 一键离线演示
└── pyproject.toml
```

---

## 核心设计

### 1. ReAct 循环的三种终止路径

一个只会 `while True` 调模型的 Agent 是不可用的。MiniAgent 显式处理三种收敛情形：

| 终止原因 | 触发条件 | 处理方式 |
|---|---|---|
| `final` | 模型输出 `Final Answer:` | 正常返回 |
| `loop_detected` | 同一动作 + 相同参数重复 > `max_repeats` 次 | 立即中止，避免无限消耗 token |
| `max_steps` | 步数用尽 | 追加「禁止再调用工具」指令，强制它基于已有观察作答 |

### 2. 解析器：宽容但不含糊

模型输出不可控，这是 Agent 线上故障的第一大来源。解析器能容忍：

- Markdown 代码块包裹（```` ```json ... ``` ````）
- 单引号 JSON、尾随逗号（`{'a': 1,}`）
- 全角冒号（`Action：`）
- 内联调用语法（`Action: calculator({"expression": "1+1"})`）
- 只给裸值不给 JSON（`Action Input: (1+2)*3`，运行时绑定到唯一参数）

同时对模糊情况保持严格：**只有 `Thought` 没有 `Action`/`Final Answer` 的输出判定为无效**，
回灌格式提示让模型自我修正，而不是把思考过程当成答案返回给用户。

### 3. 工具系统：一个装饰器搞定

```python
from miniagent.tools import tool

@tool(description="查询城市天气")
def get_weather(city: str, days: int = 1) -> str:
    """city: 城市名
    days: 预报天数"""
    return f"{city} 未来 {days} 天晴"
```

JSON Schema 由类型注解自动生成（`required` 依据默认值推断，参数说明从 docstring 提取），
执行前统一做缺参 / 多余参数 / 类型校验；工具内部抛出的任何异常都会被转换成
`Observation: 工具执行失败：...` 回灌给模型，让它有机会自行修正——而不是让整个 Agent 崩溃。

### 4. 安全边界

模型输出属于**不可信输入**，凡是它可能控制执行的地方都做了约束：

| 风险 | 防护 |
|---|---|
| 代码注入 | 计算器用 AST 白名单求值，拒绝 `eval`；拦截 `__import__`、属性访问、lambda |
| CPU 炸弹 | 指数上限 1000、节点数上限 200，挡住 `9**9**9` |
| 路径穿越 | 文件工具 `resolve()` 后校验 `relative_to(root)`，拒绝 `../`、绝对路径、盘符 |
| SQL 注入 | 全部使用 `?` 参数化查询 |
| 工具崩溃 | 执行层统一捕获，降级为观察结果 |

以上每一条都有对应的单元测试，例如 `test_reject_code_injection`、`test_path_traversal_blocked`、
`test_sql_injection_is_data`。

### 5. 可观测性

每次运行生成 `runs/run-<时间戳>.jsonl`，逐事件记录 `run_start` / `llm_call` / `step` /
`tool_result` / `run_end`，含每步耗时与原始模型输出。排查「它为什么这么做」时直接看轨迹，
不必靠猜。这批数据也可以直接作为后续评测集（eval）的来源。

---

## 测试

```bash
python -m unittest discover -s tests -t . -v     # 零依赖
python -m pytest tests -v                         # 若已安装 pytest
```

覆盖范围（68 个用例）：

- **解析器**：各种畸形输出、`Final Answer` 优先级、Observation 幻觉剔除
- **工具层**：计算器注入与炸弹、路径穿越、SQL 注入、参数校验、异常隔离
- **Agent 循环**：多步调用、死循环检测、步数耗尽收尾、模型故障降级、上下文裁剪、轨迹落盘

关键点：所有 Agent 循环测试都用 `ScriptedLLM`（按脚本返回固定文本）驱动，
**不依赖真实模型**，因此结果确定、可重复、零成本，也能在 CI 里稳定运行。

---

## 扩展：加一个自己的工具

```python
# miniagent/tools/search.py
from miniagent.tools.base import tool

@tool(description="在本地知识库里搜索关键词")
def search_notes(keyword: str) -> str:
    """keyword: 要搜索的关键词"""
    return "..."

# miniagent/tools/__init__.py 里 import 一次即可完成注册
from .search import search_notes
```

不需要改 Agent 主循环——工具清单会随注册表自动更新到提示词里。

---

## 已知限制

- 采用的是**文本协议**（Thought/Action 解析），不是原生 function calling。
  迁移到原生工具调用只需 `ToolSpec.to_openai_schema()`（已提供）替换掉提示词部分。
- 记忆是**纯上下文窗口**方案，没有向量检索式的长期记忆。
- 没有并行工具调用 / 多 Agent 协作。
- Mock 引擎只覆盖预设场景，用于演示与测试，不具备真实语言理解能力。

---

## 技术选型说明

**为什么不用 LangChain？** 因为目标是把 Agent 的运行机制弄清楚。
用框架能更快出效果，但 `AgentExecutor` 内部如何处理解析失败、如何防止死循环、
上下文怎么裁剪，都被封装掉了。手写一遍之后，这些问题的答案就变成自己的了；
之后再回到框架，也更清楚每一层抽象在替代什么。

---

## License

MIT
