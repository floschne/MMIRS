import uvicorn
from fastapi import FastAPI
from omegaconf import OmegaConf

from api.routers import general_router

# create the main app
app = FastAPI()
# include the routers
app.include_router(general_router)

if __name__ == "__main__":
    # load the config
    config = OmegaConf.load('./config.yaml')

    # read port from config
    port = config['api']['port']
    assert port is not None and isinstance(port, int), "The port has to be an integer! E.g. 8081"

    uvicorn.run(app, host="0.0.0.0", port=port, debug=True)
