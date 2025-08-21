#!/usr/bin/env python3
"""Complete setup and verification script for Codeflash MCP integration."""

import subprocess
import sys


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("🔍 Checking dependencies...")

    missing = []

    # Check core codeflash
    try:
        import codeflash

        print(f"  ✅ codeflash {codeflash.version.__version__}")
    except ImportError:
        missing.append("codeflash")
        print("  ❌ codeflash not found")

    # Check MCP dependencies
    try:
        import mcp

        print("  ✅ mcp")
    except ImportError:
        missing.append("mcp")
        print("  ❌ mcp not found")

    try:
        import fastmcp

        print("  ✅ fastmcp")
    except ImportError:
        missing.append("fastmcp")
        print("  ❌ fastmcp not found")

    try:
        import uvloop

        print("  ✅ uvloop")
    except ImportError:
        missing.append("uvloop")
        print("  ❌ uvloop not found")

    if missing:
        print(f"\n📦 Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install codeflash[mcp]")
        return False

    return True


def test_mcp_server():
    """Test that the MCP server can start."""
    print("\n🧪 Testing MCP server...")

    try:
        result = subprocess.run(
            [sys.executable, "-c", "from codeflash.mcp.server import main; print('MCP server import OK')"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        if result.returncode == 0:
            print("  ✅ MCP server imports successfully")
            return True
        print(f"  ❌ MCP server import failed: {result.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print("  ❌ MCP server test timed out")
        return False
    except Exception as e:
        print(f"  ❌ MCP server test error: {e}")
        return False


def setup_claude_code():
    """Set up Claude Code integration."""
    print("\n🔧 Setting up Claude Code integration...")

    try:
        from codeflash.mcp.integration import ClaudeCodeIntegration

        integration = ClaudeCodeIntegration()
        success, message = integration.setup_integration()

        if success:
            print(f"  ✅ {message}")
            return True
        print(f"  ⚠️  {message}")
        return False

    except Exception as e:
        print(f"  ❌ Setup failed: {e}")
        return False


def verify_integration():
    """Verify the complete integration."""
    print("\n✅ Verifying integration...")

    try:
        from codeflash.mcp.integration import ClaudeCodeIntegration

        integration = ClaudeCodeIntegration()
        status = integration.get_status()

        print(f"  Claude Code installed: {'✅' if status['claude_code_installed'] else '❌'}")
        print(f"  Config file: {status['config_file'] or 'Not found'}")
        print(f"  Codeflash configured: {'✅' if status['codeflash_configured'] else '❌'}")
        print(f"  MCP executable: {status['mcp_executable']}")

        if status["claude_code_installed"] and status["codeflash_configured"]:
            print("\n🎉 Integration is complete!")
            print("\nNext steps:")
            print("1. Restart Claude Code")
            print("2. Try: 'Show me codeflash optimization help'")
            print("3. Or: 'Optimize a function in my project using codeflash'")
            return True
        print("\n⚠️  Integration incomplete")
        return False

    except Exception as e:
        print(f"  ❌ Verification failed: {e}")
        return False


def main():
    """Main setup process."""
    print("🚀 Codeflash MCP Integration Setup")
    print("=" * 40)

    success = True

    # Step 1: Check dependencies
    if not check_dependencies():
        success = False
        print("\n❌ Please install missing dependencies first:")
        print("   pip install codeflash[mcp]")

    # Step 2: Test MCP server
    if success and not test_mcp_server():
        success = False

    # Step 3: Set up Claude Code
    if success:
        setup_claude_code()  # This can succeed partially

    # Step 4: Verify integration
    verify_integration()

    print("\n" + "=" * 40)
    if success:
        print("✅ Setup completed successfully!")
    else:
        print("⚠️  Setup completed with issues")
        print("\nFor help:")
        print("- Run: codeflash setup status")
        print("- Check: https://docs.codeflash.ai/claude-code-integration")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
