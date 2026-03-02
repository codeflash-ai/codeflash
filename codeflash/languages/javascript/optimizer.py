from __future__ import annotations

from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import logger
from codeflash.models.models import ValidCode

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash.verification.verification_utils import TestConfig


def prepare_javascript_module(
    original_module_code: str, original_module_path: Path
) -> tuple[dict[Path, ValidCode], None]:
    """Prepare a JavaScript/TypeScript module for optimization.

    Unlike Python, JS/TS doesn't need AST parsing or import analysis at this stage.
    Returns a mapping of the file path to ValidCode with the source as-is.
    """
    validated_original_code: dict[Path, ValidCode] = {
        original_module_path: ValidCode(source_code=original_module_code, normalized_code=original_module_code)
    }
    return validated_original_code, None


def verify_js_requirements(test_cfg: TestConfig) -> None:
    """Verify JavaScript/TypeScript requirements before optimization.

    Checks that Node.js, npm, and the test framework are available.
    Logs warnings if requirements are not met but does not abort.
    """
    from codeflash.languages import get_language_support
    from codeflash.languages.base import Language
    from codeflash.languages.test_framework import get_js_test_framework_or_default

    js_project_root = test_cfg.js_project_root
    if not js_project_root:
        return

    try:
        js_support = get_language_support(Language.JAVASCRIPT)
        test_framework = get_js_test_framework_or_default()
        success, errors = js_support.verify_requirements(js_project_root, test_framework)

        if not success:
            logger.warning("JavaScript requirements check found issues:")
            for error in errors:
                logger.warning(f"  - {error}")
    except Exception as e:
        logger.debug(f"Failed to verify JS requirements: {e}")
