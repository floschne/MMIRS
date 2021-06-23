from pydantic import BaseModel, Field, validator
from typing import Optional


class PSSContextRetrievalRequest(BaseModel):
    context: str = Field(description="Context sentence.")
    top_k: Optional[int] = Field(description="The number of top-k context relevant images returned", default=10)
    dataset: Optional[str] = Field(description="The dataset in which to search for the top-k images", default='coco')
    exact: Optional[bool] = Field(
        description="If true, instead of ANN search with FAISS, exact search via SBert is used to find best matching context. Takes more time!",
        default=False)

    @validator('context')
    def context_must_not_be_empty(cls, context: str):
        if context is None or len(context) == 0:
            raise ValueError(f"Context must not be empty!")
        return context.strip()

    @validator('dataset')
    def image_pool_must_exist(cls, dataset: str):
        from backend import MMIRS
        if not MMIRS.dataset_is_available(dataset):
            raise ValueError(f"Dataset {dataset} does not exist!")
        return dataset.strip()
