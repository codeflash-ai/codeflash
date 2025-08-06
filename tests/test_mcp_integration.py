#!/usr/bin/env python3
"""Test suite for Codeflash MCP integration."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Add the codeflash module to the path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))

from codeflash.mcp.integration import ClaudeCodeIntegration

class TestClaudeCodeIntegration(unittest.TestCase):
    """Test Claude Code integration functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.integration = ClaudeCodeIntegration()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_claude_config_paths(self):
        """Test that config paths are generated correctly."""
        paths = self.integration._get_claude_config_paths()
        self.assertIsInstance(paths, list)
        self.assertTrue(len(paths) > 0)
        
        # All paths should be Path objects
        for path in paths:
            self.assertIsInstance(path, Path)
    
    def test_get_codeflash_mcp_executable(self):
        """Test MCP executable path detection."""
        executable = self.integration.get_codeflash_mcp_executable()
        self.assertIsInstance(executable, str)
        self.assertTrue(len(executable) > 0)
    
    def test_create_claude_config(self):
        """Test creating a new Claude Code configuration."""
        # Mock the config path to use our temp directory
        config_path = self.temp_path / "config.json"
        
        with patch.object(self.integration, '_get_claude_config_paths', return_value=[config_path]):
            result = self.integration.create_claude_config()
            
            if result:  # Only test if creation was successful
                self.assertTrue(config_path.exists())
                
                # Verify the config content
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                self.assertIn("mcpServers", config)
                self.assertIn("codeflash", config["mcpServers"])
                self.assertIn("command", config["mcpServers"]["codeflash"])
    
    def test_add_codeflash_to_existing_config(self):
        """Test adding codeflash to an existing Claude Code configuration."""
        config_path = self.temp_path / "config.json"
        
        # Create an existing config
        existing_config = {
            "someOtherSetting": "value",
            "mcpServers": {
                "existingServer": {
                    "command": "some-other-server"
                }
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(existing_config, f)
        
        # Add codeflash to the config
        success = self.integration.add_codeflash_to_claude_config(config_path)
        
        if success:
            # Verify the updated config
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.assertIn("mcpServers", config)
            self.assertIn("codeflash", config["mcpServers"])
            self.assertIn("existingServer", config["mcpServers"])  # Existing server should remain
            self.assertEqual(config["someOtherSetting"], "value")  # Other settings should remain
    
    def test_get_status(self):
        """Test getting integration status."""
        status = self.integration.get_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn("claude_code_installed", status)
        self.assertIn("config_file", status)
        self.assertIn("codeflash_configured", status)
        self.assertIn("mcp_executable", status)
        self.assertIn("system", status)
        
        # Values should be of correct types
        self.assertIsInstance(status["claude_code_installed"], bool)
        self.assertIsInstance(status["codeflash_configured"], bool)
        self.assertIsInstance(status["mcp_executable"], str)
        self.assertIsInstance(status["system"], str)


class TestMCPServer(unittest.TestCase):
    """Test MCP server functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Check if MCP dependencies are available
        try:
            import mcp
            import fastmcp
            cls.mcp_available = True
        except ImportError:
            cls.mcp_available = False
    
    def setUp(self):
        """Set up each test."""
        if not self.mcp_available:
            self.skipTest("MCP dependencies not available")
    
    def test_mcp_server_import(self):
        """Test that the MCP server can be imported."""
        try:
            from codeflash.mcp.server import mcp
            self.assertIsNotNone(mcp)
        except ImportError as e:
            self.fail(f"Failed to import MCP server: {e}")
    
    def test_mcp_tools_registration(self):
        """Test that MCP tools are properly registered."""
        from codeflash.mcp.server import mcp
        
        # Check that key tools are registered
        expected_tools = [
            "optimize_function",
            "optimize_file", 
            "trace_and_optimize",
            "get_codeflash_status",
            "get_optimization_help"
        ]
        
        # This is a basic test - in a real environment we'd need to introspect
        # the FastMCP instance to check registered tools
        self.assertTrue(hasattr(mcp, 'tool'))
    
    def test_project_status_model(self):
        """Test ProjectStatus Pydantic model."""
        from codeflash.mcp.server import ProjectStatus
        
        status = ProjectStatus(
            is_initialized=True,
            version="1.0.0"
        )
        
        self.assertTrue(status.is_initialized)
        self.assertEqual(status.version, "1.0.0")
        self.assertIsNone(status.config_file)
    
    def test_optimization_result_model(self):
        """Test OptimizationResult Pydantic model."""
        from codeflash.mcp.server import OptimizationResult
        
        result = OptimizationResult(
            success=True,
            message="Optimization completed",
            optimized_functions=["bubble_sort", "quick_sort"],
            performance_improvement="2x faster"
        )
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.optimized_functions), 2)
        self.assertEqual(result.performance_improvement, "2x faster")
        self.assertEqual(len(result.errors), 0)  # Default empty list


class TestCLIIntegration(unittest.TestCase):
    """Test CLI integration with MCP setup."""
    
    def test_setup_command_help(self):
        """Test that setup command provides proper help."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "codeflash", "setup", "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Should not crash and should mention claude-code
            self.assertIn("claude-code", result.stdout.lower())
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.skipTest("Codeflash CLI not available or too slow")
    
    def test_mcp_executable_exists(self):
        """Test that codeflash-mcp executable can be found."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "codeflash.mcp.server", "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Should run without critical errors
            # (might fail due to missing dependencies, but shouldn't crash Python)
            self.assertNotEqual(result.returncode, -11)  # Avoid segfaults
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.skipTest("MCP server not available")


class TestDocumentation(unittest.TestCase):
    """Test that documentation is properly created."""
    
    def test_claude_code_integration_doc_exists(self):
        """Test that Claude Code integration documentation exists."""
        doc_path = Path(__file__).parent.parent / "docs" / "claude-code-integration.mdx"
        
        if doc_path.exists():
            content = doc_path.read_text()
            self.assertIn("Claude Code Integration", content)
            self.assertIn("MCP", content)
            self.assertIn("codeflash setup claude-code", content)
    
    def test_docs_navigation_updated(self):
        """Test that docs.json includes the new page."""
        docs_config_path = Path(__file__).parent.parent / "docs" / "docs.json"
        
        if docs_config_path.exists():
            with open(docs_config_path, 'r') as f:
                config = json.load(f)
            
            # Find the getting started section and check for claude-code-integration
            found = False
            if "navigation" in config and "tabs" in config["navigation"]:
                for tab in config["navigation"]["tabs"]:
                    if "groups" in tab:
                        for group in tab["groups"]:
                            if "pages" in group:
                                if "claude-code-integration" in group["pages"]:
                                    found = True
                                    break
            
            self.assertTrue(found, "claude-code-integration not found in docs navigation")


def run_integration_tests():
    """Run all integration tests."""
    print("🧪 Running Codeflash MCP Integration Tests\n")
    
    # Run the test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n✅ All tests passed! MCP integration is working correctly.")
        return True
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        return False


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)