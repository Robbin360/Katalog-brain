import asyncio
import time
from collections import defaultdict
from typing import Any, Awaitable, Callable

PROVIDER_LIMITS_RPM = {
    "gemini": 15,
    "groq": 30,
    "deepseek": 20,
    "openai": 50,
    "default": 10,
}


class ProviderRateLimiter:
    def __init__(self):
        self._calls: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    def _get_limit(self, provider: str) -> int:
        return PROVIDER_LIMITS_RPM.get(provider.lower(), PROVIDER_LIMITS_RPM["default"])

    def _clean_old_calls(self, provider: str, now: float) -> None:
        cutoff = now - 60.0
        self._calls[provider] = [t for t in self._calls[provider] if t > cutoff]

    def _get_call_count(self, provider: str) -> int:
        now = time.time()
        self._clean_old_calls(provider, now)
        return len(self._calls[provider])

    async def acquire(self, provider: str) -> None:
        while True:
            async with self._lock:
                count = self._get_call_count(provider)
                limit = self._get_limit(provider)
                if count < limit:
                    self._calls[provider].append(time.time())
                    return
                oldest_call = min(self._calls[provider])
                wait_seconds = 60.0 - (time.time() - oldest_call) + 0.1
            print(f"⏳ [RateLimiter] {provider} at limit ({count}/{limit} RPM). Waiting {wait_seconds:.1f}s...")
            await asyncio.sleep(wait_seconds)

    async def call_with_retry(
        self,
        provider: str,
        fn: Callable[[], Awaitable[Any]],
        max_retries: int = 3,
    ) -> Any:
        last_exception = None
        for attempt in range(max_retries + 1):
            await self.acquire(provider)
            try:
                result = await fn()
                return result
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                is_rate_limit = (
                    "429" in error_str
                    or "rate limit" in error_str
                    or "quota" in error_str
                    or "too many requests" in error_str
                )
                if not is_rate_limit or attempt == max_retries:
                    raise
                wait_seconds = 2 ** (attempt + 1)
                print(f"⚠️ [RateLimiter] {provider} returned 429 (attempt {attempt + 1}/{max_retries}). Backing off {wait_seconds}s...")
                await asyncio.sleep(wait_seconds)
        raise last_exception


rate_limiter = ProviderRateLimiter()


def detect_provider_from_model(model_string: str) -> str:
    model_lower = model_string.lower()
    if "gemini" in model_lower or "google" in model_lower:
        return "gemini"
    if "groq" in model_lower:
        return "groq"
    if "deepseek" in model_lower:
        return "deepseek"
    return "default"
