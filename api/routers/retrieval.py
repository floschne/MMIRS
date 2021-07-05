from fastapi import APIRouter
from loguru import logger
from starlette.responses import JSONResponse
from typing import List

from api.model import RetrievalRequest
from api.model.dataset import Dataset
from api.model.pss_context_retrieval_request import PSSContextRetrievalRequest
from api.model.pss_focus_retrieval_request import PSSFocusRetrievalRequest
from api.model.retriever import Retriever
from backend import MMIRS
from backend.util.mmirs_timer import MMIRSTimer

router = APIRouter()
timer = MMIRSTimer()

PREFIX = '/retrieval'
TAGS = ["retrieval"]


@router.post('/top_k_images',
             tags=TAGS,
             description='Retrieve the top-k images for the query.')
async def top_k_images(req: RetrievalRequest) -> JSONResponse:
    logger.info(f"POST request on {PREFIX}/top_k_images with RetrievalRequest: {req}")
    timer.start_new_timing_session()
    urls = MMIRS().retrieve_top_k_images(req)
    if req.return_timings:
        timings = timer.get_current_timing_session().get_measurements()
        return JSONResponse(content={'urls': urls, 'timings': timings})
    else:
        return JSONResponse(content=urls)


@router.post('/pss/top_k_context',
             tags=TAGS,
             description='Retrieve the top-k context related images from the PreselectionStage')
async def top_k_context(req: PSSContextRetrievalRequest) -> JSONResponse:
    logger.info(f"POST request on {PREFIX}/pss/top_k_context with req: {req}")
    timer.start_new_timing_session()
    urls = MMIRS().pss_retrieve_top_k_context_images(context=req.context,
                                                     dataset=req.dataset,
                                                     k=req.top_k,
                                                     exact=req.exact)
    if req.return_timings:
        timings = timer.get_current_timing_session().get_measurements()
        return JSONResponse(content={'urls': urls, 'timings': timings})
    else:
        return JSONResponse(content=urls)


@router.post('/pss/top_k_focus',
             tags=TAGS,
             description='Retrieve the top-k focus related images from the PreselectionStage')
async def top_k_focus(req: PSSFocusRetrievalRequest) -> JSONResponse:
    logger.info(f"POST request on {PREFIX}/pss/top_k_context with req: {req}")
    timer.start_new_timing_session()
    urls = MMIRS().pss_retrieve_top_k_focus_images(focus=req.focus,
                                                   dataset=req.dataset,
                                                   k=req.top_k,
                                                   weight_by_sim=req.weight_by_sim,
                                                   top_k_similar=req.top_k_similar_terms,
                                                   max_similar=req.max_similar_terms,
                                                   return_similar_terms=req.return_similar_terms)
    if req.return_timings:
        timings = timer.get_current_timing_session().get_measurements()
        return JSONResponse(content={'urls': urls, 'timings': timings})
    else:
        return JSONResponse(content=urls)


@router.get('/available_datasets',
            tags=TAGS,
            description='Returns the available datasets.')
async def get_available_datasets() -> List[Dataset]:
    logger.info(f"GET request on {PREFIX}/available_datasets")
    feat_pools = MMIRS().get_available_image_feature_pools()
    return [Dataset(name=fp[0], retriever_name=fp[1]) for fp in feat_pools]


@router.get('/available_retrievers',
            tags=TAGS,
            description='Returns the available retrievers.')
async def get_available_retrievers() -> List[Retriever]:
    logger.info(f"GET request on {PREFIX}/available_retrievers")
    rets = MMIRS().get_available_retrievers()
    return [Retriever(name=ret) for ret in rets]
