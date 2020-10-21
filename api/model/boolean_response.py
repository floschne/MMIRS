from pydantic import BaseModel


class BooleanResponse(BaseModel):
    value: bool
