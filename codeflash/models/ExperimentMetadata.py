from typing import Optional

from pydantic import BaseModel


class ExperimentMetadata(BaseModel):
    id: Optional[str]
    group: str
