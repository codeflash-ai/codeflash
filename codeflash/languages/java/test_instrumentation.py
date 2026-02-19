from __future__ import annotations

from typing import Any

from codeflash.languages.java.instrumentation import instrument_generated_java_test


def instrument_generated_test(*args: Any, **kwargs: Any) -> Any:
    return instrument_generated_java_test(*args, **kwargs)
