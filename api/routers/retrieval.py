import base64
from typing import Union

from fastapi import APIRouter
from starlette.responses import FileResponse

from backend import UniterImageRetriever
from logger import api_logger
from ..model import RetrievalRequest, StringResponse

router = APIRouter()

PREFIX = '/retrieval'
TAGS = ["retrieval"]

retriever = UniterImageRetriever()


@router.post("/best_matching_image",
             tags=TAGS,
             description="Retrieve the best matching (top-1) image for the query")
async def best_matching_image(req: RetrievalRequest) -> Union[FileResponse, StringResponse]:
    api_logger.info("GET request on /best_matching_image")
    req.top_k = 1  # TODO
    top_k_paths = retriever.get_top_k(req)
    if req.base64:
        with open(str(top_k_paths[0]), "rb") as img_buffer:
            return StringResponse(value=base64.b64encode(img_buffer.read()).decode('utf-8'))
    else:
        return FileResponse(str(top_k_paths[0]))
