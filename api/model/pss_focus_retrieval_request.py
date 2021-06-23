from pydantic import BaseModel, Field, validator
from typing import Optional


class PSSFocusRetrievalRequest(BaseModel):
    focus: str = Field(description="Focus word(s)")
    top_k: Optional[int] = Field(description="The number of top-k context relevant images returned", default=10)
    dataset: Optional[str] = Field(description="The dataset in which to search for the top-k images", default='coco')
    weight_by_sim: Optional[bool] = Field(
        description="If true, the focus terms are weighted by their similarity scores. Takes more time!", default=False)

    @validator('focus')
    def focus_must_not_be_empty(cls, focus: str):
        if focus is None or len(focus) == 0:
            raise ValueError(f"Focus must not be empty!")
        return focus.strip()

    @validator('dataset')
    def image_pool_must_exist(cls, dataset: str):
        from backend import MMIRS
        if not MMIRS.dataset_is_available(dataset):
            raise ValueError(f"Dataset {dataset} does not exist!")
        return dataset.strip()
