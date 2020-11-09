from typing import Optional

from pydantic import BaseModel


class RetrievalRequest(BaseModel):
    query: str
    difficult_word: Optional[str] = None
    top_k: Optional[int] = 1
