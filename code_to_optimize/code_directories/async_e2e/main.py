import time
async def fake_api_call(delay, data):
    time.sleep(0.0001)
    return f"Processed: {data}"