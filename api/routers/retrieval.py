from fastapi import APIRouter
from starlette.responses import FileResponse

from backend import UniterImageRetriever
from logger import api_logger
from ..model import RetrievalRequest

router = APIRouter()

PREFIX = '/retrieval'
TAGS = ["retrieval"]

retriever = UniterImageRetriever()


@router.post("/best_matching_image",
             tags=TAGS,
             description="Retrieve the best matching (top-1) image for the query")
async def best_matching_image(req: RetrievalRequest):
    api_logger.info("GET request on /best_matching_image")
    req.top_k = 1  # TODO
    top_k_paths = retriever.get_top_k(req)
    return FileResponse(str(top_k_paths[0]))
