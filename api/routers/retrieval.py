from fastapi import APIRouter
from starlette.responses import JSONResponse

from backend import UniterImageRetriever, LighttpImgServer
from logger import api_logger
from ..model import RetrievalRequest

router = APIRouter()

PREFIX = '/retrieval'
TAGS = ["retrieval"]

retriever = UniterImageRetriever()
image_server = LighttpImgServer()


@router.post("/best_matching_image",
             tags=TAGS,
             description="Retrieve the best matching (top-1) image for the query")
async def best_matching_image(req: RetrievalRequest) -> JSONResponse:
    api_logger.info("GET request on /best_matching_image")
    top_k_ids = retriever.get_top_k(req)
    urls = [image_server.get_img_url(img_id) for img_id in top_k_ids]
    return JSONResponse(content=urls)
