from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from enum import unique, Enum
from loguru import logger
from typing import List, Optional

from backend.fineselection.data.image_feature_pool_factory import ImageFeaturePoolFactory
from backend.fineselection.plot.max_focus_region_annotator import MaxFocusRegionAnnotator
from backend.fineselection.plot.wra_plotter import WRAPlotter
from backend.fineselection.retriever import RetrieverFactory
from backend.imgserver.py_http_image_server import PyHttpImageServer
from config import conf


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

            cls.img_server = PyHttpImageServer()

            # setup MaxFocusAnnotator
            cls.max_focus_anno = MaxFocusRegionAnnotator()

            # setup wra plotter
            cls.wra_plotter = WRAPlotter()

            # init worker pool
            cls.worker_pool = ProcessPoolExecutor(max_workers=conf.fine_selection.max_workers)

        return cls.__singleton

    def shutdown(self):
        logger.info(f'Shutting down FineSelectionStage!')
        self.worker_pool.shutdown(wait=False, cancel_futures=True)

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

        if annotate_max_focus_region or return_wra_matrices:
            wra_matrices: np.ndarray = result_dict['wra'][ranked_by.value]
            focus_span = retriever.find_focus_span_in_context(context=context, focus=focus)
            max_focus_region_indices = [retriever.find_max_focus_region_index(focus_span, wra)
                                        for iid, wra in zip(top_k_image_ids, wra_matrices)]

            futures = []
            if annotate_max_focus_region:
                futures += self.max_focus_anno.annotate_max_focus_regions(pool=self.worker_pool,
                                                                          image_ids=top_k_image_ids,
                                                                          dataset=dataset,
                                                                          max_focus_region_indices=max_focus_region_indices,
                                                                          focus_text=focus)

            if return_wra_matrices:
                # generate plots in parallel
                context_tokens = retriever.tokenize(context, remove_sep_cls=True)
                futures += self.wra_plotter.generate_wra_plots(pool=self.worker_pool,
                                                               image_ids=top_k_image_ids,
                                                               wra_matrices=wra_matrices,
                                                               max_focus_region_indices=max_focus_region_indices,
                                                               focus_span=focus_span,
                                                               context_tokens=context_tokens[0])

                for future in as_completed(futures):
                    iid, dst, task = future.result()
                    if task == 'anno':
                        # register annotated image at image server
                        self.img_server.register_annotated_image(img_id=iid,
                                                                 dataset=dataset,
                                                                 annotated_image_path=dst)
                    elif task == 'wra_plot':
                        # register wra plot at image server
                        self.img_server.register_wra_plot(img_id=iid, wra_plot_path=dst)
                    else:
                        raise ValueError(f"Task {task} is unknown!")

        return top_k_image_ids
