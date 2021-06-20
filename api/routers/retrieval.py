import time

from fastapi import APIRouter
from loguru import logger
from starlette.responses import JSONResponse

from api.model import RetrievalRequest
from backend import MMIRS

router = APIRouter()

PREFIX = '/retrieval'
TAGS = ["retrieval"]

mmirs = MMIRS()


@router.post('/top_k_images',
             tags=TAGS,
             description='Retrieve the top-k images for the query.')
async def top_k_images(req: RetrievalRequest) -> JSONResponse:
    logger.info(f"GET request on {PREFIX}/top_k_images with RetrievalRequest: {req}")
    start = time.time()
    urls = mmirs.retrieve_top_k_images(req)
    logger.info(f"MMIR execution took: {time.time() - start}")
    return JSONResponse(content=urls)
