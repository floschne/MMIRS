from loguru import logger
from typing import Tuple, Set

from backend.fineselection.data import TeranPrecomputedImageEmbeddingsPool, ImageFeaturePool
from backend.fineselection.retriever.retriever import RetrieverType
from config import conf


class ImageFeaturePoolFactory(object):
    __singleton = None

    def __new__(cls, *args, **kwargs):
        if cls.__singleton is None:
            logger.info("Instantiating ImagePoolFactory")
            cls.__singleton = super(ImageFeaturePoolFactory, cls).__new__(cls)

            cls._conf = conf.fine_selection.feature_pools

            cls.available_pools = {(source_dataset, retriever_name) for source_dataset in cls._conf.keys() for
                                   retriever_name in
                                   cls._conf[source_dataset].keys()}

            # keys -> Tuple[source_dataset, retriever_type]
            cls.pool_cache = {}

        return cls.__singleton

    def create_or_get_pool(self, source_dataset: str, retriever_name: str) -> ImageFeaturePool:
        if (source_dataset, retriever_name) not in self.available_pools:
            raise NotImplementedError(
                f"{source_dataset.upper()} ImagePool for {retriever_name.upper()} Retriever is not implemented!")

        if (source_dataset, retriever_name) in self.pool_cache:
            return self.pool_cache[(source_dataset, retriever_name)]

        pool_conf = self._conf[source_dataset][retriever_name]
        if RetrieverType.TERAN in retriever_name.lower():
            logger.info(f"Creating TeranPrecomputedImageEmbeddingsPool for {retriever_name}...")
            pool = TeranPrecomputedImageEmbeddingsPool(source_dataset=source_dataset,
                                                       pre_fetch=pool_conf.pre_fetch,
                                                       feats_root=pool_conf.feats_root,
                                                       fn_prefix=pool_conf.fn_prefix,
                                                       num_workers=pool_conf.num_workers)
            self.pool_cache[(source_dataset, retriever_name)] = pool

        elif RetrieverType.UNITER in retriever_name.lower():
            raise NotImplementedError(f"UNITER ImagePools not yet implemented!")

        else:
            raise NotImplementedError(f"ImagePools for {retriever_name.upper()} not yet implemented!")

        return self.pool_cache[(source_dataset, retriever_name)]

    def create_and_cache_all_available(self) -> None:
        for source_dataset, retriever_name in self.available_pools:
            self.create_or_get_pool(source_dataset, retriever_name)

    def get_available_pools(self) -> Set[Tuple[str, str]]:
        return self.available_pools
