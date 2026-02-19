from __future__ import annotations

from typing import TYPE_CHECKING, Any

from codeflash.languages.java.instrumentation import instrument_existing_test

if TYPE_CHECKING:
    from pathlib import Path


def inject_profiling_into_existing_java_test(
    test_string: str,
    function_to_optimize: Any,
    mode: str,
    test_path: Path | None = None,
    test_class_name: str | None = None,
) -> tuple[bool, str | None]:
    return instrument_existing_test(
        test_string=test_string,
        function_to_optimize=function_to_optimize,
        mode=mode,
        test_path=test_path,
        test_class_name=test_class_name,
    )
