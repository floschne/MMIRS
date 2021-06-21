from pydantic import BaseModel, Field


class Retriever(BaseModel):
    name: str = Field(description="Name of the retriever.")
    # type: RetrieverType = Field(description="Type of the retriever.") # TODO
