"""MiniAgent —— 一个从零实现的轻量级 ReAct 智能体框架。

核心组成：
    Agent      : 思考 → 行动 → 观察 循环调度器
    LLM        : 模型适配层（OpenAI 兼容接口 / 离线 Mock）
    tools      : 工具注册表 + 参数校验 + 安全沙箱
    memory     : 带滑动窗口裁剪的对话记忆
    Tracer     : 结构化执行轨迹记录，便于调试与复盘

设计原则：零第三方硬依赖，只用 Python 标准库即可完整运行。
"""

from .agent import Agent, AgentResult, Step
from .llm import Message, LLMError, OpenAICompatibleLLM
from .memory import ConversationMemory
from .mock_llm import MockLLM
from .tools import get_tool, list_tools, tool
from .tracer import Tracer

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentResult",
    "Step",
    "Message",
    "LLMError",
    "OpenAICompatibleLLM",
    "ConversationMemory",
    "MockLLM",
    "Tracer",
    "tool",
    "get_tool",
    "list_tools",
    "__version__",
]
