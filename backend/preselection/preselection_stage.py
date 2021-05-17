import random
import time
from enum import Enum, unique
from typing import Dict, List

from loguru import logger

from backend.preselection import ContextPreselector
from backend.preselection import FocusPreselector
from config import conf


@unique
class MergeOp(str, Enum):
    UNION = 'union'
    INTERSECTION = 'intersection'


class PreselectionStage(object):
    __singleton = None

    def __new__(cls, *args, **kwargs):
        if cls.__singleton is None:
            logger.info('Instantiating PreselectionStage!')
            cls.__singleton = super(PreselectionStage, cls).__new__(cls)

            cls.__context_preselector = ContextPreselector()
            cls.__focus_preselector = FocusPreselector()

            cls._conf = conf.preselection.stage

        return cls.__singleton

    @staticmethod
    def __merge_relevant_images(focus: Dict[str, float],
                                context: Dict[str, float],
                                max_num_relevant: int,
                                merge_op: MergeOp = MergeOp.INTERSECTION) -> List[str]:
        logger.debug(f"Merging with {merge_op}")

        if merge_op == MergeOp.UNION:
            merged = list(focus.keys()) + list(context.keys())
        elif merge_op == MergeOp.INTERSECTION:
            # intersect the key sets
            merged = list(focus.keys() & context.keys())

            # union as fallback if (way) too less items got returned
            if len(merged) < max_num_relevant // 10:
                logger.debug(f"Merging with UNION as fallback. Intersection size: {len(merged)}")
                merged = list(focus.keys()) + list(context.keys())
        else:
            raise NotImplementedError(f"Merge Operation {merge_op} not implemented!")

        if len(merged) > max_num_relevant:
            # shuffle the merged list because otherwise we would discard the docs with the lowest scores and since
            # focus relevant scores are wtf_idf scores and are larger than cosine sim scores, it would always discard
            # the focus similar docs.
            # TODO think of a way to include an equal number of context and focus relevant docs if possible.
            #  or discard context docs if more context docs are found (or vice versa for focus docs)
            #  - just take the top k//2 from context and focus !?
            random.shuffle(merged)
            return merged[:max_num_relevant]

        return merged

    def retrieve_relevant_images(self,
                                 focus: str,
                                 context: str,
                                 dataset: str,
                                 merge_op: MergeOp = MergeOp.INTERSECTION,
                                 max_num_focus_relevant: int = 5000,
                                 max_num_context_relevant: int = 5000,
                                 max_num_relevant: int = 5000,
                                 focus_weight_by_sim: bool = False,
                                 exact_context_retrieval: bool = False) -> List[str]:

        # TODO do this in two parallel threads!
        start = time.time()
        context_relevant = self.__context_preselector.retrieve_top_k_relevant_images(context,
                                                                                     k=max_num_context_relevant,
                                                                                     dataset=dataset,
                                                                                     exact=exact_context_retrieval)
        logger.info(f"ContextPreselector took: {time.time() - start}s")

        start = time.time()
        focus_relevant = self.__focus_preselector.retrieve_top_k_relevant_images(focus,
                                                                                 k=max_num_focus_relevant,
                                                                                 dataset=dataset,
                                                                                 weight_by_sim=focus_weight_by_sim)
        logger.info(f"FocusPreselector took: {time.time() - start}s")

        start = time.time()
        merged = self.__merge_relevant_images(focus=focus_relevant,
                                              context=context_relevant,
                                              max_num_relevant=max_num_relevant,
                                              merge_op=merge_op)
        logger.info(f"Merging took: {time.time() - start}s")

        return merged
