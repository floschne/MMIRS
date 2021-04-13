from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ModelChoice(str, Enum):
    TERAN = 'uniter'
    UNITER = 'teran'


class RetrievalRequest(BaseModel):
    context: str = Field(description="Query to the image retrieval model. E.g. 'Two girls enjoying some icecream.'")
    focus: Optional[str] = Field(description="Focus word(s) in the context. E.g. 'icecream'")
    top_k: Optional[int] = 3
    model: Optional[ModelChoice] = 'teran'
