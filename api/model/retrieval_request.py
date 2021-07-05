from pydantic import BaseModel, Field, validator, root_validator
from typing import Optional

from backend.fineselection.fine_selection_stage import RankedBy


class RetrievalRequest(BaseModel):
    context: str = Field(description="Query to the image retrieval model. E.g. 'Two girls enjoying some icecream.'")
    focus: str = Field(description="Focus word(s) in the context. E.g. 'icecream'")
    top_k: Optional[int] = Field(description="The number of returned top-k images!", default=10)
    retriever: Optional[str] = Field(description="Retriever (model) that performs the retrieval task",
                                     default='teran_coco')
    # TODO allow multiple datasets --> make union and search on union
    dataset: Optional[str] = Field(description="The dataset in which the Retriever searches for the top-k images",
                                   default='coco')
    annotate_max_focus_region: Optional[bool] = Field(
        description="If true, the region with the maximum focus signal is highlighted in the returned images",
        default=False)
    ranked_by: Optional[RankedBy] = Field(description="Method to rank the images. Either by context, focus or combined",
                                          default=RankedBy.COMBINED)
    focus_weight: Optional[float] = Field(
        description="Weight of the focus when ranking the images (w * focus + (1-2) * context)",
        default=0.5)
    return_scores: Optional[bool] = Field(description="If true the scores of the top-k images are returned",
                                          default=False)
    return_wra_matrices: Optional[bool] = Field(description="If true the WRA matrices of the top-k images are returned",
                                                default=False)
    return_timings: Optional[bool] = Field(description="If true timings of the operation are returned",
                                           default=False)

    @root_validator
    def focus_must_exist_in_context(cls, values):
        context = values.get('context')
        focus = values.get('focus')

        if focus is None or len(focus) == 0 or focus not in context:
            raise ValueError(f"Focus must not be empty and must exist in context!")
        return values

    @validator('retriever')
    def retriever_must_exist(cls, retriever: str):
        from backend import MMIRS
        if retriever.strip() not in MMIRS.get_available_retrievers():
            raise ValueError(f"Retriever {retriever} does not exist!")
        return retriever.strip()

    @validator('dataset')
    def image_pool_must_exist(cls, dataset: str):
        from backend import MMIRS
        if not MMIRS.dataset_is_available(dataset):
            raise ValueError(f"Dataset {dataset} does not exist!")
        return dataset.strip()

    @validator('ranked_by')
    def ranked_by_must_be_valid(cls, ranked_by: RankedBy):
        if ranked_by != RankedBy.COMBINED or ranked_by != RankedBy.FOCUS or ranked_by != RankedBy.COMBINED:
            ranked_by = RankedBy.COMBINED
        return ranked_by

    @validator('focus_weight')
    def focus_weight_must_be_in_range(cls, focus_weight: float):
        if focus_weight < 0. or focus_weight > 1.:
            raise ValueError("Focus Weight has to be between 0 and 1!")
        return focus_weight

    @validator('context')
    def context_must_not_be_empty(cls, context: str):
        if context is None or len(context) == 0:
            raise ValueError("Context must not be empty!")
        # fixme hacky
        if context[-1] != '.':
            context += '.'
        return context