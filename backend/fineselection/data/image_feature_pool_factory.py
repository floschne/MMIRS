from typing import Tuple, Set

from loguru import logger
from omegaconf import OmegaConf

from backend.fineselection.data import TeranImageFeaturePool, ImageFeaturePool
from backend.fineselection.retriever.retriever import RetrieverType


class ImageFeaturePoolFactory(object):
    __singleton = None

    def __new__(cls, *args, **kwargs):
        if cls.__singleton is None:
            logger.info("Instantiating ImagePoolFactory")
            cls.__singleton = super(ImageFeaturePoolFactory, cls).__new__(cls)

            cls._conf = OmegaConf.load("config.yaml").fine_selection.feature_pools

            cls.available_pools = {(source_dataset, retriever_type) for source_dataset in cls._conf.keys() for
                                   retriever_type in
                                   cls._conf[source_dataset].keys()}

            # keys -> Tuple[source_dataset, retriever_type]
            cls.pool_cache = {}

        return cls.__singleton

    def create_or_get_pool(self, source_dataset: str, retriever_type: str) -> ImageFeaturePool:
        if (source_dataset, retriever_type) not in self.available_pools:
            raise NotImplementedError(
                f"{source_dataset.upper()} ImagePool for {retriever_type.upper()} Retriever is not implemented!")

        if (source_dataset, retriever_type) in self.pool_cache:
            return self.pool_cache[(source_dataset, retriever_type)]

        pool_conf = self._conf[source_dataset][retriever_type]
        if retriever_type.lower() == RetrieverType.TERAN:
            self.pool_cache[(source_dataset, retriever_type)] = TeranImageFeaturePool(source_dataset=source_dataset,
                                                                                      pre_fetch=pool_conf.pre_fetch,
                                                                                      data_root=pool_conf.data_root,
                                                                                      fn_prefix=pool_conf.fn_prefix,
                                                                                      num_workers=pool_conf.num_workers)
        elif retriever_type.lower() == RetrieverType.UNITER:
            raise NotImplementedError(f"UNITER ImagePools not yet implemented!")

        else:
            raise NotImplementedError(f"ImagePools for {retriever_type.upper()} not yet implemented!")

        return self.pool_cache[(source_dataset, retriever_type)]

    def create_and_cache_all_available(self) -> None:
        for source_dataset, retriever_type in self.available_pools:
            self.create_or_get_pool(source_dataset, retriever_type)

    def get_available_pools(self) -> Set[Tuple[str, str]]:
        return self.available_pools