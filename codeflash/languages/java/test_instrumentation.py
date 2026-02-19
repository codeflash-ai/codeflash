from __future__ import annotations

from codeflash.languages.java.instrumentation import instrument_generated_java_test


def instrument_generated_test(*args, **kwargs):
    return instrument_generated_java_test(*args, **kwargs)
