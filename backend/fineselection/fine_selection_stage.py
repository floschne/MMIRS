from loguru import logger
from typing import List, Optional

from backend.fineselection.annotator.max_focus_region_annotator import MaxFocusRegionAnnotator
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

            # setup MaxFocusAnnotator
            cls.max_focus_anno = MaxFocusRegionAnnotator()

        return cls.__singleton

    def find_top_k_images(self,
                          focus: Optional[str],  # TODO what should happen if focus is None further down
                          context: str,
                          top_k: int,
                          retriever_name: str,
                          dataset: str,
                          preselected_image_ids: List[str] = None,
                          annotate_max_focus_region: bool = False):
        # get the retriever
        retriever = self.retriever_factory.create_or_get_retriever(retriever_name)

        # get the image pool for the selected dataset and retriever
        pool = self.pool_factory.create_or_get_pool(source_dataset=dataset, retriever_type=retriever.retriever_type)

        # build and load the image search space (into memory!) containing the preselected images
        iss = pool.get_image_search_space(preselected_image_ids)

        # do the retrieval
        top_k_image_ids, alignment_matrices = retriever.find_top_k_images(focus=focus,
                                                                          context=context,
                                                                          top_k=top_k,
                                                                          iss=iss,
                                                                          return_alignment_matrices=True)
        if annotate_max_focus_region:
            annotated_paths = [self.max_focus_anno.annotate_max_focus_region(image_id=iid,
                                                                             dataset=dataset,
                                                                             wra_matrix=alignment_matrices,
                                                                             focus=focus,
                                                                             context=context)
                               for iid in top_k_image_ids]
        return top_k_image_ids
