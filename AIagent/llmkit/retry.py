"""重试 + 指数退避，供优化课演示"健壮性调优"。

真实网络会抖动、会 429、会 5xx。生产 Agent 必须能优雅重试。
这里给出一个不依赖任何库、带 jitter 的指数退避实现。
"""

from __future__ import annotations

import random
import time
from typing import Callable, Tuple, Type, TypeVar

T = TypeVar("T")


def with_retry(
    fn: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retry_on: Tuple[Type[Exception], ...] = (Exception,),
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """执行 fn，失败按指数退避重试。

    退避公式：delay = min(max_delay, base_delay * 2**attempt) + 随机 jitter。
    jitter 很关键——避免大量客户端在同一时刻重试造成"惊群/雷鸣羊群"。
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except retry_on as e:
            last_exc = e
            if attempt == max_retries:
                break
            delay = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0, base_delay)
            sleep(delay)
    assert last_exc is not None
    raise last_exc
