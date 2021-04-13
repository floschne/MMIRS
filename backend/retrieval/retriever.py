from abc import abstractmethod
from pathlib import Path
from typing import List, Union

from loguru import logger
from omegaconf import OmegaConf

from api.model import RetrievalRequest


class Retriever(object):
    def __init__(self, retriever_name: str):
        logger.info(f"Instantiating {retriever_name} Retriever...")
        self._conf = OmegaConf.load("config.yaml").retriever[retriever_name]

    @abstractmethod
    def get_top_k(self, req: RetrievalRequest) -> List[Union[Path, str]]:
        """
        :param req: the request
        :return: Path to or IDs of the image(s)
        """
        pass
