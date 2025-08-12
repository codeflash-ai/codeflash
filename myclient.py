import asyncio
from fastmcp import Client

client = Client("myserver.py")

async def call_tool(file: str, function: str)-> None:
    async with client:
        result = await client.call_tool("optimize_code", {"file": f"{file}", "function":f"{function}"})
        print(result)

asyncio.run(call_tool("Ford","Mustang"))
