import subprocess
import sys


def test_help_displays_logo():
    result = subprocess.run(
        [sys.executable, "-c", "from codeflash.main import main; main()", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "codeflash.ai" in result.stdout


def test_help_short_flag_displays_logo():
    result = subprocess.run(
        [sys.executable, "-c", "from codeflash.main import main; main()", "-h"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "codeflash.ai" in result.stdout
