from pydantic import BaseModel


class StringResponse(BaseModel):
    value: str
