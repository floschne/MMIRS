from typing import List, Tuple, Set

from loguru import logger
from omegaconf import OmegaConf

from backend import LighttpImgServer
from backend.fineselection import FineSelectionStage
from backend.fineselection.data import ImageFeaturePoolFactory
from backend.fineselection.retriever import RetrieverFactory
from backend.preselection import PreselectionStage


class MMIRS(object):
    __singleton = None

    def __new__(cls, *args, **kwargs):
        if cls.__singleton is None:
            logger.info('Instantiating MMIRS!')
            cls.__singleton = super(MMIRS, cls).__new__(cls)

            cls._conf = OmegaConf.load("config.yaml").mmirs

            cls.pss = PreselectionStage()
            cls.fss = FineSelectionStage()

            cls.img_srv = LighttpImgServer()

        return cls.__singleton

    def retrieve_top_k_images(self,
                              focus: str,
                              context: str,
                              top_k: int,
                              retriever_name: str,
                              dataset: str) -> List[str]:
        """
        Retrieves the top-k matching images according to focus and context in the specified image pool with the specified
        retriever.

        :param focus:
        :type focus:
        :param context:
        :type context:
        :param top_k:
        :type top_k:
        :param retriever_name:
        :type retriever_name:
        :param dataset:
        :type dataset:
        :return: URLs of the top-k matching images
        :rtype:
        """
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
                                                   preselected_image_ids=pss_imgs)

        # get URLs
        top_k_img_urls = self.img_srv.get_img_urls(top_k_img_ids, dataset)
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