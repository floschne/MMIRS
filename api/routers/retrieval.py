from fastapi import APIRouter
from loguru import logger
from starlette.responses import JSONResponse

from backend.retrieval import UniterImageRetriever, TeranImageRetriever
from backend import LighttpImgServer
from api.model import RetrievalRequest

router = APIRouter()

PREFIX = '/retrieval'
TAGS = ["retrieval"]

retrievers = {'uniter': UniterImageRetriever(),
              'teran': TeranImageRetriever()}
image_server = LighttpImgServer()


@router.post("/best_matching_image",
             tags=TAGS,
             description="Retrieve the best matching (top-1) image for the query.")
async def best_matching_image(req: RetrievalRequest) -> JSONResponse:
    logger.info(f"GET request on {PREFIX}/best_matching_image with RetrievalRequest: {req}")
    top_k_ids = retrievers[req.model].get_top_k(req)
    urls = [image_server.get_img_url(img_id, req.model) for img_id in top_k_ids]
    return JSONResponse(content=urls)


@router.post('/top_k_images',
             tags=TAGS,
             description='Retrieve the top-k images for the query.')
async def top_k_images(req: RetrievalRequest) -> JSONResponse:
    logger.info(f"GET request on {PREFIX}/top_k_images with RetrievalRequest: {req}")
    top_k_ids = retrievers[req.model].get_top_k(req)
    urls = [image_server.get_img_url(img_id, req.model) for img_id in top_k_ids]
    return JSONResponse(content=urls)
