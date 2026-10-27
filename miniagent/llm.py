"""模型适配层。

对外只暴露一个极简契约：``chat(messages) -> str``。
这样上层 Agent 循环完全不需要关心背后是 GPT、DeepSeek 还是本地 Qwen，
也方便在测试里塞一个假的模型做纯逻辑验证（依赖倒置）。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


@dataclass(slots=True)
class Message:
    """一条对话消息。role 取值：system / user / assistant。"""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    def __str__(self) -> str:  # 便于日志打印
        return f"[{self.role}] {self.content}"


class LLMError(RuntimeError):
    """模型调用失败（网络、鉴权、限流等）。"""


@runtime_checkable
class LLM(Protocol):
    """模型协议。任何实现该方法的对象都能驱动 Agent。"""

    model_name: str

    def chat(
        self, messages: Sequence[Message], stop: Sequence[str] | None = None
    ) -> str:  # pragma: no cover - 协议定义
        ...


_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OpenAICompatibleLLM:
    """通过 HTTP 直连任意 OpenAI 兼容的 /chat/completions 接口。

    用标准库 urllib 而非第三方 SDK，避免环境依赖问题；
    想换 SDK 时只要保持 chat() 签名不变，上层无需改动。

        支持的供应商：OpenAI / DeepSeek / 阿里百炼 / 智谱 / vLLM / Ollama ...
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self.model_name = model or os.getenv("MINIAGENT_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

    # ------------------------------------------------------------------ 公开方法
    def chat(self, messages: Sequence[Message], stop: Sequence[str] | None = None) -> str:
        if not self.api_key:
            raise LLMError(
                "未配置 API Key。请设置环境变量 OPENAI_API_KEY，"
                "或使用 --provider mock 以离线模式运行。"
            )

        payload: dict[str, object] = {
            "model": self.model_name,
            "messages": [m.to_dict() for m in messages],
            "temperature": self.temperature,
        }
        if stop:
            payload["stop"] = list(stop)

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        url = f"{self.base_url}/chat/completions"

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = self._post(url, body)
                return self._extract_text(raw)
            except urllib.error.HTTPError as exc:
                detail = _safe_read(exc)
                last_err = LLMError(f"HTTP {exc.code}: {detail}")
                if exc.code not in _RETRYABLE_STATUS or attempt == self.max_retries:
                    raise last_err from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_err = LLMError(f"网络错误: {exc}")
                if attempt == self.max_retries:
                    raise last_err from exc
            except LLMError:
                raise

            # 指数退避：0.8s, 1.6s ...
            time.sleep(0.8 * (2**attempt))

        raise last_err or LLMError("未知错误")

    # ------------------------------------------------------------------ 内部实现
    def _post(self, url: str, body: bytes) -> dict:
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _extract_text(data: dict) -> str:
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"响应格式异常: {json.dumps(data, ensure_ascii=False)[:300]}") from exc


def _safe_read(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", "replace")[:300]
    except Exception:  # noqa: BLE001 - 错误信息读取失败不应掩盖原始异常
        return str(exc)
