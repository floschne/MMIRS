import time

from fastapi import APIRouter
from loguru import logger
from starlette.responses import JSONResponse
from typing import List

from api.model import RetrievalRequest
from api.model.dataset import Dataset
from api.model.retriever import Retriever
from backend import MMIRS
from backend.fineselection.retriever.retriever import RetrieverType

router = APIRouter()

PREFIX = '/retrieval'
TAGS = ["retrieval"]

mmirs = MMIRS()


@router.post('/top_k_images',
             tags=TAGS,
             description='Retrieve the top-k images for the query.')
async def top_k_images(req: RetrievalRequest) -> JSONResponse:
    logger.info(f"POST request on {PREFIX}/top_k_images with RetrievalRequest: {req}")
    start = time.time()
    urls = mmirs.retrieve_top_k_images(req)
    logger.info(f"MMIR execution took: {time.time() - start}")
    return JSONResponse(content=urls)


@router.get('/available_datasets',
            tags=TAGS,
            description='Returns the available datasets.')
async def get_available_datasets() -> List[Dataset]:
    logger.info(f"GET request on {PREFIX}/available_datasets")
    feat_pools = mmirs.get_available_image_feature_pools()
    return [Dataset(name=fp[0], retriever_type=fp[1]) for fp in feat_pools]


@router.get('/available_retrievers',
            tags=TAGS,
            description='Returns the available retrievers.')
async def get_available_datasets() -> List[Retriever]:
    logger.info(f"GET request on {PREFIX}/available_retrievers")
    rets = mmirs.get_available_retrievers()
    return [Retriever(name=ret) for ret in rets]
