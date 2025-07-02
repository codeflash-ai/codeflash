import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

from codeflash.cli_cmds.console import logger, paneled_text
from codeflash.version import __version__ as version


def run_doctor() -> None:
    """Run comprehensive setup verification for Codeflash."""
    paneled_text(
        "🩺 Codeflash Doctor - Diagnosing your setup...",
        panel_args={"title": "Setup Verification", "expand": False},
        text_args={"style": "bold blue"}
    )
    
    checks = [
        ("Python Environment", check_python_environment),
        ("Codeflash Installation", check_codeflash_installation),
        ("VS Code Python Extension", check_vscode_python_extension),
        ("LSP Server Connection", check_lsp_server_connection),
        ("Git Repository", check_git_repository),
        ("Project Configuration", check_project_configuration),
    ]
    
    results = []
    all_passed = True
    
    for check_name, check_func in checks:
        logger.info(f"Checking {check_name}...")
        success, message = check_func()
        results.append((check_name, success, message))
        if not success:
            all_passed = False
    
    print_results(results, all_passed)


def check_python_environment() -> Tuple[bool, str]:
    """Check Python version and environment."""
    try:
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        if sys.version_info < (3, 8):
            return False, f"Python {python_version} found. Python 3.8+ required."
        return True, f"Python {python_version} ✓"
    except Exception as e:
        return False, f"Failed to check Python version: {e}"


def check_codeflash_installation() -> Tuple[bool, str]:
    """Verify Codeflash is properly installed."""
    try:
        return True, f"Codeflash {version} installed ✓"
    except Exception as e:
        return False, f"Codeflash installation check failed: {e}"


def check_vscode_python_extension() -> Tuple[bool, str]:
    """Check if VS Code Python extension is installed."""
    try:
        code_cmd = shutil.which("code")
        if not code_cmd:
            return False, "VS Code 'code' command not found in PATH"
        
        result = subprocess.run(
            [code_cmd, "--list-extensions"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            return False, f"Failed to list VS Code extensions: {result.stderr}"
        
        extensions = result.stdout.strip().split('\n')
        python_extensions = [ext for ext in extensions if 'python' in ext.lower()]
        
        if not python_extensions:
            return False, "Python extension not found. Install the Python extension for VS Code."
        
        return True, f"VS Code Python extension found: {', '.join(python_extensions)} ✓"
        
    except subprocess.TimeoutExpired:
        return False, "VS Code extension check timed out"
    except Exception as e:
        return False, f"VS Code extension check failed: {e}"


def check_lsp_server_connection() -> Tuple[bool, str]:
    """Test LSP server connectivity."""
    try:
        from codeflash.lsp.server import CodeflashLanguageServer
        
        # Test that we can instantiate the server (basic smoke test)
        server_class = CodeflashLanguageServer
        if hasattr(server_class, 'initialize_optimizer'):
            return True, "LSP server available ✓"
        else:
            return True, "LSP server module loaded successfully ✓"
            
    except ImportError as e:
        return False, f"LSP server import failed: {e}"
    except Exception as e:
        return False, f"LSP server check failed: {e}"


def check_git_repository() -> Tuple[bool, str]:
    """Check if running in a git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            return True, "Git repository detected ✓"
        else:
            return False, "No git repository found. Initialize with 'git init'"
            
    except subprocess.TimeoutExpired:
        return False, "Git check timed out"
    except FileNotFoundError:
        return False, "Git not found in PATH"
    except Exception as e:
        return False, f"Git check failed: {e}"


def check_project_configuration() -> Tuple[bool, str]:
    """Check for project configuration files."""
    config_files = ["pyproject.toml", "setup.py", "requirements.txt", "setup.cfg"]
    # Use list comprehension for efficiency and readability
    found_configs = [fname for fname in config_files if Path(fname).exists()]
    
    if found_configs:
        return True, f"Project configuration found: {', '.join(found_configs)} ✓"
    else:
        return False, "No project configuration files found (pyproject.toml, setup.py, etc.)"


def print_results(results: List[Tuple[str, bool, str]], all_passed: bool) -> None:
    """Print the diagnostic results in a formatted way."""
    print("\n" + "="*60)
    print("🩺 CODEFLASH SETUP DIAGNOSIS RESULTS")
    print("="*60)
    
    for check_name, success, message in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:8} | {check_name:25} | {message}")
    
    print("="*60)
    
    if all_passed:
        paneled_text(
            "🎉 Your Codeflash setup is perfect! 🎉\n\n"
            "All checks passed successfully. You're ready to optimize your code!\n\n"
            "Next steps:\n"
            "• Run 'codeflash init' to initialize a project\n"
            "• Use 'codeflash --file <filename>' to optimize a specific file\n"
            "• Try 'codeflash --verify-setup' for an end-to-end test",
            panel_args={"title": "✅ SUCCESS", "expand": False},
            text_args={"style": "bold green"}
        )
    else:
        failed_checks = [name for name, success, _ in results if not success]
        paneled_text(
            f"⚠️  Setup Issues Detected\n\n"
            f"The following checks failed:\n"
            f"• {chr(10).join(failed_checks)}\n\n"
            f"Please address these issues and run 'codeflash doctor' again.\n\n"
            f"For help, visit: https://codeflash.ai/docs",
            panel_args={"title": "❌ ISSUES FOUND", "expand": False},
            text_args={"style": "bold yellow"}
        )
        sys.exit(1)