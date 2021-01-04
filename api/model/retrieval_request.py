from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ModelChoice(str, Enum):
    TERAN = 'uniter'
    UNITER = 'teran'


class RetrievalRequest(BaseModel):
    query: str = Field(description="Query to the image retrieval model. E.g. 'Two girls enjoying some icecream.'")
    difficult_word: Optional[str] = None
    top_k: Optional[int] = 3
    model: Optional[ModelChoice] = 'teran'
