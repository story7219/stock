import asyncio
import time

class RateLimiter:
    """
    초당 최대 N건의 API 호출만 허용하는 Throttle 유틸리티
    """
    def __init__(self, max_calls_per_sec: int):
        self.max_calls = max_calls_per_sec
        self.call_times = []

    async def acquire(self):
        now = time.monotonic()
        self.call_times = [t for t in self.call_times if now - t < 1.0]
        if len(self.call_times) >= self.max_calls:
            sleep_time = 1.0 - (now - self.call_times[0])
            await asyncio.sleep(max(sleep_time, 0))
        self.call_times.append(time.monotonic()) 