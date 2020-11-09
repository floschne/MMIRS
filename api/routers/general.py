from fastapi import APIRouter
from fastapi.responses import RedirectResponse

from logger import api_logger
from ..model import BooleanResponse, StringResponse

router = APIRouter()


@router.get("/", tags=["general"], description="Redirection to /docs")
async def root_to_docs():
    api_logger.info("GET request on / -> redirecting to /docs")
    return RedirectResponse("/docs")


@router.get("/heartbeat", response_model=BooleanResponse, tags=["general"], description="Heartbeat check")
async def heartbeat():
    api_logger.info("GET request on /heartbeat")
    return BooleanResponse(value=True)


@router.get("/hello", response_model=StringResponse, tags=["general"], description="Redirection to /docs")
async def root_to_docs():
    api_logger.info("GET request on /hello")
    return StringResponse(value="Hello World!")
