"""飞书机器人通知。"""

from __future__ import annotations

import json
from urllib import error, request


FEISHU_WEBHOOK_URL = (
    "https://open.feishu.cn/open-apis/bot/v2/hook/"
    "b56e5960-c6e8-4ab9-a009-f751b4af9b4a"
)


def send_feishu_message(message: str, timeout: float = 10.0) -> bool:
    """发送文本消息；失败时打印原因但不抛出异常。"""
    payload = json.dumps(
        {"msg_type": "text", "content": {"text": message}},
        ensure_ascii=False,
    ).encode("utf-8")
    req = request.Request(
        FEISHU_WEBHOOK_URL,
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"[飞书通知] 发送失败: {exc}")
        return False

    if result.get("code", result.get("StatusCode", 0)) != 0:
        print(f"[飞书通知] 发送失败: {result}")
        return False

    print("[飞书通知] 发送成功")
    return True
