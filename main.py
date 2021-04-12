import uvicorn
from fastapi import FastAPI
from omegaconf import OmegaConf

import api.routers.general as general
import api.routers.retrieval as retrieval

# create the main app
app = FastAPI()

# include the routers
app.include_router(general.router)
app.include_router(retrieval.router, prefix=retrieval.PREFIX)

if __name__ == "__main__":
    # load the config
    config = OmegaConf.load('./config.yaml')

    # read port from config
    port = config['api']['port']
    assert port is not None and isinstance(port, int), "The port has to be an integer! E.g. 8081"

    uvicorn.run(app, host="0.0.0.0", port=port, debug=True)
