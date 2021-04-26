import random
from enum import Enum, unique
from typing import Dict, List

from loguru import logger
from omegaconf import OmegaConf

from backend.preselection import ContextPreselector
from backend.preselection import FocusPreselector


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

            cls._conf = OmegaConf.load('config.yaml').preselection.stage

        return cls.__singleton

    @staticmethod
    def __merge_relevant_images(focus: Dict[str, float],
                                context: Dict[str, float],
                                merge_op: MergeOp = MergeOp.INTERSECTION) -> List[str]:
        merged = {}
        if merge_op == MergeOp.UNION:
            merged.update(focus)
            merged.update(context)
        if merge_op == MergeOp.INTERSECTION:
            # intersect the key sets
            intersect = focus.keys() & context.keys()
            # take the max relevance score
            # actually this makes not too much sense because focus scores are wtf_idf and context scores are cosine sims
            # but we'll discard the scores anyways later
            merged = {k: max(focus[k], context[k]) for k in intersect}

        # merged = {k: v for k, v in sorted(merged.items(), key=lambda i: i[1], reverse=True)}
        return list(merged.keys())

    def retrieve_relevant_images(self,
                                 focus: str,
                                 context: str,
                                 dataset: str,
                                 merge_op: MergeOp = MergeOp.INTERSECTION,
                                 max_num_relevant: int = 5000,
                                 focus_weight_by_sim: bool = False,
                                 exact_context_retrieval: bool = False) -> List[str]:

        # TODO do this in two parallel threads!
        context_relevant = self.__context_preselector.retrieve_top_k_relevant_images(context,
                                                                                     k=max_num_relevant,
                                                                                     dataset=dataset,
                                                                                     exact=exact_context_retrieval)
        focus_relevant = self.__focus_preselector.retrieve_top_k_relevant_images(focus,
                                                                                 k=max_num_relevant,
                                                                                 dataset=dataset,
                                                                                 weight_by_sim=focus_weight_by_sim)

        merged = self.__merge_relevant_images(focus_relevant, context_relevant, merge_op)
        if merge_op == MergeOp.INTERSECTION and len(merged) < max_num_relevant // 10:
            # switch merge_op to union as fallback if (way) too less items got returned
            merged = self.__merge_relevant_images(focus_relevant, context_relevant, MergeOp.UNION)

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
