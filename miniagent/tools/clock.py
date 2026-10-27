"""时钟工具：让 Agent 具备「知道现在几点」的能力。

看似简单，却是 Agent 必备的上下文补全——大模型不知道当前时间，
涉及「今天」「明天」「本周」的推理全靠它。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .base import tool

_WEEKDAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


@tool(description="获取当前日期时间，可指定时区偏移（小时）")
def now(utc_offset: int = 8) -> str:
    """utc_offset: 相对 UTC 的时区偏移小时数，默认 8（东八区）"""
    tz = timezone(timedelta(hours=utc_offset))
    current = datetime.now(tz)
    weekday = _WEEKDAYS[current.weekday()]
    return f"{current.strftime('%Y-%m-%d %H:%M:%S')} {weekday} (UTC{utc_offset:+d})"
