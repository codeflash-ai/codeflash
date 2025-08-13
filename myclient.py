import anthropic
from rich import print
url = 'https://0de03d07f8b9.ngrok-free.app'
client = anthropic.Anthropic()
response = client.beta.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Optimize my codebase, the file is \"/Users/codeflash/Downloads/codeflash-dev/codeflash/code_to_optimize/bubble_sort.py\" and the function is \"sorter\""}],
    mcp_servers=[
        {
            "type": "url",
            "url": f"{url}/mcp/",
            "name": "Code Optimization Assistant",
        }
    ],
    extra_headers={
        "anthropic-beta": "mcp-client-2025-04-04"
    }
)
print(response.content)