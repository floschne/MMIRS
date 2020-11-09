from abc import abstractmethod
from pathlib import Path

from omegaconf import OmegaConf

from api.model import RetrievalRequest
from backend.util import SingletonABCMeta
from logger import backend_logger


class Retriever(object):
    __metaclass__ = SingletonABCMeta

    def __init__(self, retriever_name: str):
        backend_logger.info(f"Instantiating {retriever_name} Retriever...")
        self._conf = OmegaConf.load("config.yaml").retriever[retriever_name]

    @abstractmethod
    def get_top_k(self, req: RetrievalRequest) -> Path:
        """
        :param req: the request
        :return: Path to the image(s)
        """
        pass
