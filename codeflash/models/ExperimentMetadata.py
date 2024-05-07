from __future__ import annotations

from pydantic import BaseModel


class ExperimentMetadata(BaseModel):
    id: str
    group: str
