from __future__ import annotations

from codeflash.languages.java.instrumentation import instrument_existing_test


def inject_profiling_into_existing_java_test(*args, **kwargs):
    return instrument_existing_test(*args, **kwargs)
