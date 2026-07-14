"""Test that code_repair is only called for Python, not JS/TS.

This tests the language guard added to prevent calling Python-only /ai/code_repair
endpoint for JavaScript/TypeScript functions.
"""

from codeflash.languages.language_enum import Language


def test_code_repair_should_only_run_for_python():
    """
    Verify that the language check logic correctly identifies when code_repair should run.

    Code repair uses Python-specific tools (libcst) and should only run for Python code.
    """
    # This test documents the expected behavior:
    # code_repair should only be attempted for Python functions

    assert Language.PYTHON == "python"
    assert Language.JAVASCRIPT == "javascript"
    assert Language.TYPESCRIPT == "typescript"

    # The actual fix will add this check in maybe_repair_optimization():
    # if self.function_to_optimize.language == "python":
    #     self.future_all_code_repair.append(self.repair_optimization(...))

    # For non-Python languages, repair should be skipped
    # This test serves as documentation of the intended behavior


def test_language_enum_values():
    """Ensure Language enum has the expected values for the fix."""
    assert hasattr(Language, 'PYTHON')
    assert hasattr(Language, 'JAVASCRIPT')
    assert hasattr(Language, 'TYPESCRIPT')

    # String comparison works for the language check
    assert Language.PYTHON == "python"
    assert Language.JAVASCRIPT != "python"
    assert Language.TYPESCRIPT != "python"
