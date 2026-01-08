import asyncio


async def retry_with_backoff(func, max_retries=3):
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")
    last_exception = None
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                await asyncio.sleep(0.0001 * attempt)
    raise last_exception
