from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock

from codeflash.discovery.functions_to_optimize import was_function_previously_optimized


def test_was_function_previously_optimized_ignores_missing_git_remote(monkeypatch) -> None:
    function_to_optimize = Mock(file_path=Path("example.py"), qualified_name="sorter")
    code_context = Mock(hashing_code_context_hash="hash-123")
    check_optimization_status = Mock()

    monkeypatch.setattr("codeflash.discovery.functions_to_optimize.is_LSP_enabled", Mock(return_value=False))
    monkeypatch.setattr("codeflash.discovery.functions_to_optimize.is_subagent_mode", Mock(return_value=False))
    monkeypatch.setattr("codeflash.discovery.functions_to_optimize.get_pr_number", Mock(return_value=123))
    monkeypatch.setattr(
        "codeflash.discovery.functions_to_optimize.get_repo_owner_and_name",
        Mock(side_effect=ValueError("Remote named 'origin' didn't exist")),
    )
    monkeypatch.setattr(
        "codeflash.discovery.functions_to_optimize.is_function_being_optimized_again",
        check_optimization_status,
    )

    result = was_function_previously_optimized(function_to_optimize, code_context, Namespace(no_pr=False))

    assert result is False
    check_optimization_status.assert_not_called()
