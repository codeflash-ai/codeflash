import asyncio
async def fake_api_call(delay, data):
    await asyncio.sleep(0.0001)
    return f"Processed: {data}"


async def some_api_call(urls):
    results = []
    for url in urls:
        res = await fake_api_call(1, url)
        results.append(res)
    return results