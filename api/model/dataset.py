from pydantic import BaseModel, Field

from backend.fineselection.retriever.retriever import RetrieverType


class Dataset(BaseModel):
    name: str = Field(description="Name of the dataset.")
    retriever_name: str = Field(description="Name of the retriever that can search through the dataset.")
    # number_of_images: int = Field(description="Number of images contained in the dataset.") # TODO
