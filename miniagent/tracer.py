"""执行轨迹记录（Trace）。

Agent 的调试痛点是「它为什么这么做」。把每一步的思考、动作、观察、耗时
落成 JSONL，既能事后复盘，也能作为评测集（eval）的数据来源。

事件类型：
    run_start / llm_call / step / tool_result / run_end / error
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any


class Tracer:
    """轻量结构化日志。线程不安全（单 Agent 单线程足够用）。"""

    def __init__(self, path: str | Path | None = None, enabled: bool = True) -> None:
        self.enabled = enabled
        self.events: list[dict[str, Any]] = []
        self._started = time.perf_counter()

        if not enabled:
            self.path: Path | None = None
            return

        if path is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = Path("runs") / f"run-{stamp}.jsonl"
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    # ------------------------------------------------------------------ API
    def event(self, kind: str, **data: Any) -> None:
        if not self.enabled:
            return

        record = {
            "ts": round(datetime.now().timestamp(), 3),
            "elapsed_ms": round((time.perf_counter() - self._started) * 1000, 1),
            "event": kind,
            **data,
        }
        self.events.append(record)

        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    @property
    def path_str(self) -> str | None:
        return str(self.path) if self.path else None

    def summary(self) -> dict[str, int]:
        """按事件类型统计，方便 README 里展示。"""
        counts: dict[str, int] = {}
        for e in self.events:
            counts[e["event"]] = counts.get(e["event"], 0) + 1
        return counts
