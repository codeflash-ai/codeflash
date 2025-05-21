from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import dataclasses


@dataclasses.dataclass
class FunctionModules:
    function_name: str
    file_name: Path
    module_name: str
    class_name: Optional[str] = None
    line_no: Optional[int] = None
