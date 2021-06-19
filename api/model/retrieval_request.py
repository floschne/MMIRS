from typing import Optional

from pydantic import BaseModel, Field, validator, root_validator


class RetrievalRequest(BaseModel):
    context: str = Field(description="Query to the image retrieval model. E.g. 'Two girls enjoying some icecream.'")
    focus: Optional[str] = Field(description="Focus word(s) in the context. E.g. 'icecream'")
    top_k: Optional[int] = Field(description="Focus word(s) in the context. E.g. 'icecream'", default=10)
    retriever: Optional[str] = Field(description="Retriever (model) that performs the retrieval task",
                                     default='teran_coco')
    # TODO allow multiple datasets --> make union and search on union
    dataset: Optional[str] = Field(description="The dataset in which the Retriever searches for the top-k images",
                                   default='coco')

    @root_validator
    def non_empty_focus_must_exist_in_context(cls, values):
        context = values.get('context')
        focus = values.get('focus')

        if (focus is not None and len(focus) != 0) and focus not in context:
            raise ValueError(f"Non empty focus must exist in context!")
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
