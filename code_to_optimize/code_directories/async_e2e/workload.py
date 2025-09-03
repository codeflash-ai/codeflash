import asyncio

async def fake_api_call(delay, data):
    await asyncio.sleep(0.0001)
    return f"Processed: {data}"
