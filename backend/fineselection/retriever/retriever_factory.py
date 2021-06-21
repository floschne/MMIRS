from loguru import logger
from typing import List

from backend.fineselection.retriever import Retriever, TeranRetriever, UniterRetriever
from config import conf


class RetrieverFactory(object):
    __singleton = None

    def __new__(cls, *args, **kwargs):
        if cls.__singleton is None:
            logger.info("Instantiating RetrieverFactory")
            cls.__singleton = super(RetrieverFactory, cls).__new__(cls)

            cls._conf = conf.fine_selection.retrievers

            # TODO also store the type of the retrieve instead of just the name
            cls.available_retrievers = cls._conf.keys()
            cls.retriever_cache = {}

        return cls.__singleton

    def create_or_get_retriever(self, retriever_name: str) -> Retriever:
        if retriever_name not in self.available_retrievers:
            raise NotImplementedError(f"Retriever with name {retriever_name} is not implemented!")

        if retriever_name in self.retriever_cache:
            return self.retriever_cache[retriever_name]

        retriever_conf = self._conf[retriever_name]

        if retriever_conf.retriever_type.lower() == 'teran':
            self.retriever_cache[retriever_name] = TeranRetriever(retriever_name=retriever_name,
                                                                  device=retriever_conf.device,
                                                                  model=retriever_conf.model,
                                                                  model_config=retriever_conf.model_config)
            return self.retriever_cache[retriever_name]
        elif retriever_conf.retriever_type.lower() == 'uniter':
            self.retriever_cache[retriever_name] = UniterRetriever(retriever_name=retriever_name,
                                                                   n_gpu=retriever_conf.n_gpu,
                                                                   uniter_dir=retriever_conf.uniter_dir,
                                                                   model_config=retriever_conf.model_config,
                                                                   num_imgs=retriever_conf.num_imgs,
                                                                   batch_size=retriever_conf.batch_size,
                                                                   n_data_workers=retriever_conf.n_data_workers,
                                                                   fp16=retriever_conf.fp16,
                                                                   pin_mem=retriever_conf.pin_mem)
        else:
            raise NotImplementedError(f"Retrievers of type {retriever_conf.retriever_type} not implemented!")

        return self.retriever_cache[retriever_name]

    def create_and_cache_all_available(self):
        for ret in self.available_retrievers:
            self.create_or_get_retriever(ret)

    def get_available_retrievers(self) -> List[str]:
        return list(self.available_retrievers)
