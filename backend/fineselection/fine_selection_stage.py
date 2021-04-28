from typing import List

from loguru import logger

from backend.fineselection.data.image_feature_pool_factory import ImageFeaturePoolFactory
from backend.fineselection.retriever import RetrieverFactory


class FineSelectionStage(object):
    __singleton = None

    def __new__(cls, *args, **kwargs):
        if cls.__singleton is None:
            cls.__singleton = super(FineSelectionStage, cls).__new__(cls)
            logger.info("Instantiating Fine Selection Stage...")

            # setup and build retrievers
            cls.retriever_factory = RetrieverFactory()
            cls.retriever_factory.create_and_cache_all_available()

            # setup and build image pools
            cls.pool_factory = ImageFeaturePoolFactory()
            cls.pool_factory.create_and_cache_all_available()

        return cls.__singleton

    def find_top_k_images(self,
                          focus: str,
                          context: str,
                          top_k: int,
                          retriever_name: str,
                          dataset: str,
                          preselected_image_ids: List[str] = None):
        # get the retriever
        retriever = self.retriever_factory.create_or_get_retriever(retriever_name)

        # get the image pool for the selected dataset and retriever
        pool = self.pool_factory.create_or_get_pool(source_dataset=dataset, retriever_type=retriever.retriever_type)

        # build and load the image search space (into memory!) containing the preselected images
        iss = pool.get_image_search_space(preselected_image_ids)

        # do the retrieval
        top_k_image_ids = retriever.find_top_k_images(focus, context, top_k, iss)
        
        return top_k_image_ids
