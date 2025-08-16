import asyncio


async def async_add(a, b):
    """Simple async function that adds two numbers."""
    await asyncio.sleep(0.001)  # Simulate some async work
    print(f"codeflash stdout: Adding {a} + {b}")
    result = a + b
    print(f"result: {result}")
    return result
