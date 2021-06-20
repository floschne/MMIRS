import numpy as np
from enum import unique, Enum
from loguru import logger
from typing import List, Optional

from backend.fineselection.data.image_feature_pool_factory import ImageFeaturePoolFactory
from backend.fineselection.plot.max_focus_region_annotator import MaxFocusRegionAnnotator
from backend.fineselection.plot.wra_plotter import WRAPlotter
from backend.fineselection.retriever import RetrieverFactory


@unique
class RankedBy(str, Enum):
    FOCUS = 'focus'
    CONTEXT = 'context'
    COMBINED = 'combined'


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

            # setup wra plotter
            cls.wra_plotter = WRAPlotter()

        return cls.__singleton

    def find_top_k_images(self,
                          focus: Optional[str],  # TODO what should happen if focus is None further down
                          context: str,
                          top_k: int,
                          retriever_name: str,
                          dataset: str,
                          preselected_image_ids: List[str] = None,
                          ranked_by: RankedBy = RankedBy.COMBINED,
                          annotate_max_focus_region: bool = False,
                          focus_weight: float = 0.5,
                          return_scores: bool = False,
                          return_wra_matrices: bool = False):

        # get the retriever
        retriever = self.retriever_factory.create_or_get_retriever(retriever_name)

        # get the image pool for the selected dataset and retriever
        pool = self.pool_factory.create_or_get_pool(source_dataset=dataset, retriever_type=retriever.retriever_type)

        # build and load the image search space (into memory!) containing the preselected images
        iss = pool.get_image_search_space(preselected_image_ids)

        return_separated_ranks = False
        if ranked_by != RankedBy.COMBINED:
            return_separated_ranks = True

        # do the retrieval
        result_dict = retriever.find_top_k_images(focus=focus,
                                                  context=context,
                                                  top_k=top_k,
                                                  iss=iss,
                                                  focus_weight=focus_weight,
                                                  return_scores=return_scores,
                                                  return_wra_matrices=return_wra_matrices or annotate_max_focus_region,
                                                  return_separated_ranks=return_separated_ranks or annotate_max_focus_region)

        top_k_image_ids = result_dict['top_k'][ranked_by.value]

        if annotate_max_focus_region:
            focus_span = retriever.find_focus_span_in_context(context=context, focus=focus)
            wra_matrices: np.ndarray = result_dict['wra'][ranked_by.value]

            for iid, wra in zip(top_k_image_ids, wra_matrices):
                max_region_idx = retriever.find_max_focus_region_index(focus_span, wra)
                self.max_focus_anno.annotate_max_focus_region(image_id=iid,
                                                              dataset=dataset,
                                                              max_region_idx=max_region_idx,
                                                              focus_text=focus)
        if return_wra_matrices:
            wra_matrices: np.ndarray = result_dict['wra'][ranked_by.value]

            for iid, wra in zip(top_k_image_ids, wra_matrices):
                context_tokens = retriever.tokenize(context, remove_sep_cls=True)
                self.wra_plotter.plot_wra(image_id=iid,
                                          context_tokens=context_tokens[0],
                                          wra=wra)
        return top_k_image_ids
