from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from git import Repo

from codeflash.cli_cmds.github_workflow import install_github_actions


def test_install_github_actions_skip_confirm_uses_defaults_without_prompts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    Repo.init(tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        """
[tool.codeflash]
module-root = "."
tests-root = "tests"
formatter-cmds = ["disabled"]
""".strip(),
        encoding="utf-8",
    )

    with (
        patch("codeflash.cli_cmds.github_workflow.inquirer.prompt") as mock_prompt,
        patch("codeflash.cli_cmds.github_workflow.get_current_branch", return_value="main"),
        patch("codeflash.cli_cmds.github_workflow.get_repo_owner_and_name", side_effect=RuntimeError("no remote")),
    ):
        install_github_actions(skip_confirm=True)

    mock_prompt.assert_not_called()
    assert (tmp_path / ".github" / "workflows" / "codeflash.yaml").exists()


def test_install_github_actions_skip_confirm_supports_go_projects(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    Repo.init(tmp_path)
    (tmp_path / "go.mod").write_text("module example.com/demo\n\ngo 1.21\n", encoding="utf-8")

    with (
        patch("codeflash.cli_cmds.github_workflow.inquirer.prompt") as mock_prompt,
        patch("codeflash.cli_cmds.github_workflow.get_current_branch", return_value="main"),
        patch("codeflash.cli_cmds.github_workflow.get_repo_owner_and_name", side_effect=RuntimeError("no remote")),
    ):
        install_github_actions(skip_confirm=True)

    workflow_path = tmp_path / ".github" / "workflows" / "codeflash.yaml"
    workflow_text = workflow_path.read_text(encoding="utf-8")

    mock_prompt.assert_not_called()
    assert workflow_path.exists()
    assert "Optimize new Go code" in workflow_text
    assert "actions/setup-go@v5" in workflow_text
    assert "go mod download" in workflow_text


def test_install_github_actions_non_tty_skips_optional_setup(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    Repo.init(tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        """
[tool.codeflash]
module-root = "."
tests-root = "tests"
formatter-cmds = ["disabled"]
""".strip(),
        encoding="utf-8",
    )

    stdin = type("Stdin", (), {"isatty": lambda self: False})()
    monkeypatch.setattr("codeflash.cli_cmds.github_workflow.sys.stdin", stdin)

    with patch("codeflash.cli_cmds.github_workflow.inquirer.prompt") as mock_prompt:
        install_github_actions()

    mock_prompt.assert_not_called()
    assert not (tmp_path / ".github" / "workflows" / "codeflash.yaml").exists()
