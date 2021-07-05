import uvicorn
from fastapi import FastAPI
from loguru import logger

import api.routers.general as general
import api.routers.retrieval as retrieval
from backend import MMIRS
from backend.fineselection import FineSelectionStage
from backend.imgserver.py_http_image_server import PyHttpImageServer
from backend.util.mmirs_timer import MMIRSTimer
from config import conf

# create the main api
app = FastAPI(title="User Study API",
              description="Simple API that powers my Master Thesis' user study.",
              version="beta")


@logger.catch(reraise=True)
@app.on_event("shutdown")
def shutdown_event():
    try:
        PyHttpImageServer().shutdown()
        FineSelectionStage().shutdown()
    except:
        pass


@logger.catch(reraise=True)
@app.on_event("startup")
def startup_event():
    # setup logger
    logger.add('logs/{time}.log', rotation=f"{conf.logging.max_file_size} MB", level=conf.logging.level.upper())

    # start MMIRS
    MMIRS()

    # start timer service
    MMIRSTimer()


# include the routers
app.include_router(general.router)
app.include_router(retrieval.router, prefix=retrieval.PREFIX)

if __name__ == "__main__":
    # read port from config
    port = conf['api']['port']
    assert port is not None and isinstance(port, int), "The port has to be an integer! E.g. 8081"

    uvicorn.run(app, host="0.0.0.0", port=port, debug=True)
