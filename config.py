import os
from loguru import logger

from omegaconf import OmegaConf

# global config
__conf_file__ = os.getenv("MMIRS_CONFIG", "configs/ltgpu1_config.yaml")
logger.info(f"Loading config '{__conf_file__}'")

conf = OmegaConf.load(__conf_file__)
