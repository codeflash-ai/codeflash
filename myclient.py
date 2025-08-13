# import asyncio
# from fastmcp import Client
#
# client = Client("myserver.py")
#
# async def call_tool(file: str, function: str)-> None:
#     async with client:
#         result = await client.call_tool("optimize_code", {"file": f"{file}", "function":f"{function}"})
#         print(result)
#
# asyncio.run(call_tool("Ford","Mustang"))


import anthropic
from rich import print

# Your server URL (replace with your actual URL)
url = 'https://0de03d07f8b9.ngrok-free.app'

client = anthropic.Anthropic()

#file and fn names are hard coded for now
response = client.beta.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Optimize my codebase, the file is \"/Users/codeflash/Downloads/codeflash-dev/codeflash/code_to_optimize/bubble_sort.py\" and the function is \"sorter\""}],
    mcp_servers=[
        {
            "type": "url",
            "url": f"{url}/mcp/",
            "name": "HelpfulAssistant",
        }
    ],
    extra_headers={
        "anthropic-beta": "mcp-client-2025-04-04"
    }
)

print(response.content)


# import anthropic
# from rich import print
#
# # Your server URL (replace with your actual URL)
# url = 'https://0de03d07f8b9.ngrok-free.app'
#
# client = anthropic.Anthropic()
#
# response = client.beta.messages.create(
#     model="claude-sonnet-4-20250514",
#     max_tokens=1000,
#     messages=[{"role": "user", "content": "Roll a few dice!"}],
#     mcp_servers=[
#         {
#             "type": "url",
#             "url": f"{url}/mcp/",
#             "name": "Dice Roller",
#         }
#     ],
#     extra_headers={
#         "anthropic-beta": "mcp-client-2025-04-04"
#     }
# )
#
# print(response.content)