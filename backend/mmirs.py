from loguru import logger
from typing import List, Tuple, Set, Union

# from api.model import RetrievalRequest
from backend.fineselection import FineSelectionStage
from backend.fineselection.data import ImageFeaturePoolFactory
from backend.fineselection.retriever import RetrieverFactory
from backend.imgserver.py_http_image_server import PyHttpImageServer
from backend.preselection import PreselectionStage
from config import conf


class MMIRS(object):
    __singleton = None

    def __new__(cls, *args, **kwargs):
        if cls.__singleton is None:
            logger.info('Instantiating MMIRS!')
            cls.__singleton = super(MMIRS, cls).__new__(cls)

            cls._conf = conf.mmirs

            if cls._conf.img_server == 'pyhttp':
                cls.img_srv = PyHttpImageServer()
            else:
                raise NotImplementedError(f"Image Server {cls._conf.img_server} not available!")

            cls.pss = PreselectionStage()
            cls.fss = FineSelectionStage()

        return cls.__singleton

    # FIXME we cannot give a type hint for req: RetrievalRequest b
    def retrieve_top_k_images(self, req) -> Union[List[str], Tuple[List[str], List[str]]]:
        """
        Retrieves the top-k matching images according to focus and context in the specified image pool with the specified
        retriever.
        """

        focus = req.focus
        context = req.context
        top_k = req.top_k
        retriever_name = req.retriever
        dataset = req.dataset
        annotate_max_focus_region = req.annotate_max_focus_region

        ranked_by = req.ranked_by
        focus_weight = req.focus_weight
        return_scores = req.return_scores
        return_wra_matrices = req.return_wra_matrices

        # find relevant images via PreselectionStage
        pss_imgs = self.pss.retrieve_relevant_images(focus=focus,
                                                     context=context,
                                                     dataset=dataset,
                                                     merge_op=self._conf.pss.merge_op,
                                                     max_num_focus_relevant=self._conf.pss.max_num_focus_relevant,
                                                     max_num_context_relevant=self._conf.pss.max_num_context_relevant,
                                                     max_num_relevant=self._conf.pss.max_num_relevant,
                                                     focus_weight_by_sim=self._conf.pss.focus_weight_by_sim,
                                                     exact_context_retrieval=self._conf.pss.exact_context_retrieval)

        # find the top-k images in the relevant images via FineSelectionStage
        top_k_img_ids = self.fss.find_top_k_images(focus=focus,
                                                   context=context,
                                                   top_k=top_k,
                                                   retriever_name=retriever_name,
                                                   dataset=dataset,
                                                   preselected_image_ids=pss_imgs,
                                                   ranked_by=ranked_by,
                                                   annotate_max_focus_region=annotate_max_focus_region,
                                                   focus_weight=focus_weight,
                                                   return_scores=return_scores,
                                                   return_wra_matrices=return_wra_matrices)

        # get URLs
        top_k_img_urls = self.img_srv.get_img_urls(top_k_img_ids, dataset, annotated=annotate_max_focus_region)
        if return_wra_matrices:
            top_k_wra_urls = self.img_srv.get_wra_urls(top_k_img_ids)
            return top_k_img_urls, top_k_wra_urls
        return top_k_img_urls

    @staticmethod
    def get_available_image_feature_pools() -> Set[Tuple[str, str]]:
        return ImageFeaturePoolFactory().get_available_pools()

    @staticmethod
    def dataset_is_available(dataset: str) -> bool:
        pools = MMIRS.get_available_image_feature_pools()
        for ds, target_ret in pools:
            if ds == dataset:
                return True
        return False

    @staticmethod
    def get_available_retrievers() -> List[str]:
        return RetrieverFactory().get_available_retrievers()
