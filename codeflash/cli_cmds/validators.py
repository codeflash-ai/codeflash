from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual.validation import Validator

from codeflash.code_utils.code_utils import validate_relative_directory_path

if TYPE_CHECKING:
    from textual.validation import ValidationResult


class PathExistsValidator(Validator):
    def __init__(self, failure_description: str | None = None) -> None:
        super().__init__(failure_description=failure_description or "Path must exist and be a directory")

    def validate(self, value: str) -> ValidationResult:
        if not value:
            return self.failure("Path cannot be empty")

        path = Path(value)
        if not path.exists():
            return self.failure(f"Path does not exist: {value}")

        if not path.is_dir():
            return self.failure(f"Path is not a directory: {value}")

        return self.success()


class RelativePathValidator(Validator):
    def __init__(self, failure_description: str | None = None) -> None:
        super().__init__(failure_description=failure_description or "Must be a valid relative directory path")

    def validate(self, value: str) -> ValidationResult:
        if not value:
            return self.failure("Path cannot be empty")

        is_valid, error_msg = validate_relative_directory_path(value)
        if not is_valid:
            return self.failure(error_msg)

        return self.success()


class APIKeyValidator(Validator):
    def __init__(self, failure_description: str | None = None) -> None:
        super().__init__(
            failure_description=failure_description or "API key must start with 'cf-' prefix. Please try again."
        )

    def validate(self, value: str) -> ValidationResult:
        # Allow empty for "press enter to open browser" flow
        if not value:
            return self.success()

        if not value.startswith("cf-"):
            return self.failure(f"That key [{value}] seems to be invalid. It should start with a 'cf-' prefix.")

        return self.success()


class NotEqualPathValidator(Validator):
    """Validator to ensure a path is not equal to another path."""

    def __init__(self, exclude_path: Path | str, failure_description: str | None = None) -> None:
        self.exclude_path = Path(exclude_path).resolve()
        super().__init__(
            failure_description=failure_description
            or f"Path cannot be the same as {exclude_path}. Please choose a different directory."
        )

    def validate(self, value: str) -> ValidationResult:
        if not value:
            return self.success()

        try:
            input_path = (Path.cwd() / Path(value)).resolve()
            if input_path == self.exclude_path:
                return self.failure(
                    f"Tests root cannot be the same as module root ({self.exclude_path}). "
                    "This can lead to unexpected behavior."
                )
        except Exception:  # noqa: S110
            # If path resolution fails, let other validators handle it
            pass

        return self.success()


class DirectoryOrCreatableValidator(Validator):
    """Validator that accepts existing directories or creates them if they don't exist."""

    def __init__(self, create_if_missing: bool = True, failure_description: str | None = None) -> None:
        self.create_if_missing = create_if_missing
        super().__init__(
            failure_description=failure_description or "Must be a valid directory path or creatable location"
        )

    def validate(self, value: str) -> ValidationResult:
        if not value:
            return self.failure("Path cannot be empty")

        path = Path(value)

        # Check if it exists and is a directory
        if path.exists():
            if path.is_dir():
                return self.success()
            return self.failure(f"Path exists but is not a directory: {value}")

        # Path doesn't exist - check if parent exists and is writable
        if not self.create_if_missing:
            return self.failure(f"Directory does not exist: {value}")

        parent = path.parent
        if not parent.exists():
            return self.failure(f"Parent directory does not exist: {parent}")

        if not parent.is_dir():
            return self.failure(f"Parent path is not a directory: {parent}")

        # Check if we can write to parent (would be able to create directory)
        import os

        if not os.access(parent, os.W_OK):
            return self.failure(f"Cannot create directory - parent is not writable: {parent}")

        return self.success()


class TomlFileValidator(Validator):
    """Validator to ensure path is a valid .toml file."""

    def __init__(self, failure_description: str | None = None) -> None:
        super().__init__(failure_description=failure_description or "Must be a valid .toml file")

    def validate(self, value: str) -> ValidationResult:
        if not value:
            return self.failure("Path cannot be empty")

        path = Path(value)

        if not path.exists():
            return self.failure(f"Configuration file not found: {value}")

        if not path.is_file():
            return self.failure(f"Configuration file is not a file: {value}")

        if path.suffix != ".toml":
            return self.failure(f"Configuration file is not a .toml file: {value}")

        return self.success()


class PyprojectTomlValidator(Validator):
    """Validator for pyproject.toml with codeflash configuration."""

    def __init__(self, failure_description: str | None = None) -> None:
        super().__init__(failure_description=failure_description or "Invalid pyproject.toml configuration")

    def validate(self, value: str) -> ValidationResult:
        if not value:
            return self.failure("Path cannot be empty")

        path = Path(value)

        # First check if it's a valid toml file
        if not path.exists():
            return self.failure(f"Configuration file not found: {value}")

        if not path.is_file():
            return self.failure(f"Path is not a file: {value}")

        if path.suffix != ".toml":
            return self.failure(f"File is not a .toml file: {value}")

        # Parse and validate codeflash configuration
        try:
            from codeflash.code_utils.config_parser import parse_config_file

            config, _ = parse_config_file(path)
        except Exception as e:
            return self.failure(f"Failed to parse configuration: {e}")

        # Validate module_root
        module_root = config.get("module_root")
        if not module_root:
            return self.failure("Missing required field: 'module_root'")

        if not Path(module_root).is_dir():
            return self.failure(f"Invalid 'module_root': directory does not exist at {module_root}")

        # Validate tests_root
        tests_root = config.get("tests_root")
        if not tests_root:
            return self.failure("Missing required field: 'tests_root'")

        if not Path(tests_root).is_dir():
            return self.failure(f"Invalid 'tests_root': directory does not exist at {tests_root}")

        return self.success()
