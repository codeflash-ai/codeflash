# Codeflash LSP and MCP Server

This directory contains the Codeflash server implementation that supports both Language Server Protocol (LSP) and Model Context Protocol (MCP).

## Features

### LSP Server
- Provides VS Code extension integration
- Custom features for function optimization
- Real-time code analysis and optimization suggestions

### MCP Server  
- Exposes Codeflash optimization capabilities as MCP tools
- Provides file access resources
- Compatible with any MCP client (Claude Desktop, etc.)

## MCP Tools

### `optimize_code(file: str, function: str)`
Optimize a specific function in a Python file using Codeflash AI optimization.

**Parameters:**
- `file`: Path to the Python file containing the function
- `function`: Name of the function to optimize

**Returns:** Dictionary with optimization results, speedup metrics, and optimized code

### `get_optimizable_functions(file: str)`
Get a list of functions that can be optimized in a file.

**Parameters:**
- `file`: Path to the Python file to analyze

**Returns:** Dictionary with list of optimizable function names

### `set_api_key(api_key: str)`
Set Codeflash API key for optimization services.

**Parameters:**
- `api_key`: Codeflash API key (should start with 'cf-')

**Returns:** Status and message

## MCP Resources

### `file://{path}`
Access file content by path.

### `codeflash://functions/{file_path}`
Get optimizable functions in a file as JSON resource.

## Usage

### Running as MCP Server

```bash
# Using the dedicated MCP entry point
python -m codeflash.lsp.mcp_entry

# Using the dual server
python -m codeflash.lsp.dual_server --mode mcp
```

### Running as LSP Server

```bash
# Using the dual server
python -m codeflash.lsp.dual_server --mode lsp

# Using the original LSP entry point
python -m codeflash.lsp.server_entry
```

### MCP Client Configuration

For Claude Desktop or other MCP clients, add this to your MCP configuration:

```json
{
  "mcpServers": {
    "codeflash": {
      "command": "python",
      "args": ["-m", "codeflash.lsp.mcp_entry"],
      "env": {
        "CODEFLASH_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

## Testing

Run the test suite to validate functionality:

```bash
python -m codeflash.lsp.test_mcp
```

## Architecture

The implementation reuses the existing LSP server infrastructure and adds MCP capabilities through the FastMCP library. Both protocols can run simultaneously, allowing the same server instance to serve both VS Code extensions and MCP clients.

Key components:
- `server.py`: Main server class with both LSP and MCP functionality
- `mcp_entry.py`: Dedicated MCP server entry point
- `dual_server.py`: Unified entry point for both protocols
- `beta.py`: LSP-specific features and handlers
- `test_mcp.py`: Test suite for MCP functionality