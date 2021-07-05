from pydantic import BaseModel, Field, validator
from typing import Optional


class PSSFocusRetrievalRequest(BaseModel):
    focus: str = Field(description="Focus word(s)")
    top_k: Optional[int] = Field(description="The number of top-k context relevant images returned", default=10)
    dataset: Optional[str] = Field(description="The dataset in which to search for the top-k images", default='coco')
    weight_by_sim: Optional[bool] = Field(
        description="If true, the focus terms are weighted by their similarity scores. Takes more time!", default=False)
    top_k_similar_terms: Optional[int] = Field(
        description="The number top-k similar focus terms used to search through the VWTF-IDF indices", default=None)
    max_similar_terms: Optional[int] = Field(
        description="The number maximum similar focus terms used to search through the VWTF-IDF indices",
        default=None)
    return_similar_terms: Optional[bool] = Field(description="If true, the similar focus terms are returned!",
                                                 default=False)
    return_timings: Optional[bool] = Field(description="If true timings of the operation are returned",
                                           default=False)

    @validator('focus')
    def focus_must_not_be_empty(cls, focus: str):
        if focus is None or len(focus) == 0:
            raise ValueError(f"Focus must not be empty!")
        return focus.strip()

    @validator('max_similar_terms')
    def max_similar_terms_must_be_larger_zero(cls, max_similar_terms: int):
        if max_similar_terms <= 0:
            raise ValueError(f"Max similar terms must be larger than 0!")
        return max_similar_terms

    @validator('top_k_similar_terms')
    def top_k_similar_terms_must_be_larger_zero(cls, top_k_similar_terms: int):
        if top_k_similar_terms <= 0:
            raise ValueError(f"Top-k similar terms must be larger than 0!")
        return top_k_similar_terms

    @validator('dataset')
    def image_pool_must_exist(cls, dataset: str):
        from backend import MMIRS
        if not MMIRS.dataset_is_available(dataset):
            raise ValueError(f"Dataset {dataset} does not exist!")
        return dataset.strip()
