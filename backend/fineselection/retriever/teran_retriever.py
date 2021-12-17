import os
import sys
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import numpy as np
from loguru import logger
from sklearn.preprocessing import minmax_scale
from typing import Tuple, Dict, List, Union

from backend.fineselection.data import TeranISS
from backend.fineselection.retriever import Retriever
from backend.fineselection.retriever.retriever import RetrieverType
from backend.util.mmirs_timer import MMIRSTimer

TERAN_PATH = 'models/teran'
sys.path.append(TERAN_PATH)

# noinspection PyUnresolvedReferences
from inference import prepare_model_checkpoint_and_config, load_teran, compute_distances, QueryEncoder, get_tokenizer


class TeranRetriever(Retriever):
    def __init__(self, retriever_name: str, device: str, model: str, model_config: str):
        super().__init__(retriever_type=RetrieverType.TERAN,
                         retriever_name=retriever_name)

        opts = self.__build_retrieval_opts(device, model, model_config)
        logger.debug(opts)

        # load teran config and checkpoint
        teran_config, model_checkpoint = prepare_model_checkpoint_and_config(opts)
        # load TERAN
        logger.info(f"Loading TERAN model {opts.model}!")
        teran = load_teran(teran_config, model_checkpoint)

        # load the tokenizer from TERAN
        self.tokenizer = get_tokenizer(teran_config)

        self.dist_pool = ProcessPoolExecutor(max_workers=32)

        self.teran = teran
        self.model_config = teran_config

        self.query_encoder = QueryEncoder(self.model_config, self.teran)

        self.timer = MMIRSTimer()

    @logger.catch
    def find_top_k_images(self,
                          focus: str,
                          context: str,
                          top_k: int,
                          iss: TeranISS,
                          focus_weight: float = 0.5,
                          return_scores: bool = False,
                          return_wra_matrices: bool = False,
                          return_separated_ranks: bool = False) -> Dict[str, Dict[str, Union[List[str], np.ndarray]]]:
        self.timer.start_measurement("TeranRetriever::find_top_k_images")
        self.timer.start_measurement("TeranRetriever::find_top_k_images::compute_query_embedding")
        # compute query embedding
        query_embs, query_lengths = self.query_encoder.compute_query_embedding(context)
        self.timer.stop_measurement()

        # get the precomputed image embeddings and lengths tensors to compute the similarity with the query embedding
        img_embs, img_length = iss.get_images()

        # compute the matching scores
        wra_matrices: np.ndarray  # type hint. shape: (k, num_roi, num_tok)
        global_similarity_scores: np.ndarray  # type hint. shape: (k)

        if focus is None:  # should only ever happen for l2 eval..
            global_similarity_scores = compute_distances(img_embs,
                                                         query_embs,
                                                         img_length,
                                                         query_lengths,
                                                         self.model_config,
                                                         return_wra_matrices=False,
                                                         dist_pool=self.dist_pool)
            context_sorted_indices = np.argsort(global_similarity_scores)[::-1][:top_k]
            return iss.get_image_ids(context_sorted_indices)

        self.timer.start_measurement("TeranRetriever::find_top_k_images::compute_distances")
        global_similarity_scores, wra_matrices = compute_distances(img_embs,
                                                                   query_embs,
                                                                   img_length,
                                                                   query_lengths,
                                                                   self.model_config,
                                                                   return_wra_matrices=True)
        self.timer.stop_measurement()

        # compute the matching scores wrt the focus
        focus_scores = self.compute_focus_scores(focus=focus,
                                                 context=context,
                                                 wra_matrices=wra_matrices,
                                                 focus_pooling='avg')

        # compute the combined scores
        combined_scores = self.compute_combined_scores(global_scores=global_similarity_scores,
                                                       focus_scores=focus_scores,
                                                       alpha=focus_weight)

        # argsort to get the indices of the images
        self.timer.start_measurement("TeranRetriever::find_top_k_images::sort_scores")
        context_sorted_indices = np.argsort(global_similarity_scores)[::-1][:top_k]
        focus_sorted_indices = np.argsort(focus_scores)[::-1][:top_k]
        combined_sorted_indices = np.argsort(combined_scores)[::-1][:top_k]
        self.timer.stop_measurement()

        # get the ranked image ids
        self.timer.start_measurement("TeranRetriever::find_top_k_images::build_return_dict")
        context_ranked_top_k: List[str] = iss.get_image_ids(context_sorted_indices)
        focus_ranked_top_k: List[str] = iss.get_image_ids(focus_sorted_indices)
        combined_ranked_top_k: List[str] = iss.get_image_ids(combined_sorted_indices)

        return_dict = {'top_k': {'combined': combined_ranked_top_k}}

        if return_separated_ranks:
            return_dict['top_k']['context'] = context_ranked_top_k
            return_dict['top_k']['focus'] = focus_ranked_top_k

        if return_scores:
            return_dict['scores'] = {'combined': combined_scores[combined_sorted_indices]}
            if return_separated_ranks:
                return_dict['scores']['context'] = global_similarity_scores[context_sorted_indices]
                return_dict['scores']['focus'] = focus_scores[focus_sorted_indices]

        if return_wra_matrices:
            return_dict['wra'] = {'combined': wra_matrices[combined_sorted_indices, ...]}
            if return_separated_ranks:
                return_dict['wra']['context'] = wra_matrices[context_sorted_indices, ...]
                return_dict['wra']['focus'] = wra_matrices[focus_sorted_indices, ...]
        self.timer.stop_measurement()

        self.timer.stop_measurement()
        return return_dict

    def compute_combined_scores(self,
                                global_scores: np.ndarray,
                                focus_scores: np.ndarray,
                                alpha: float = 0.5) -> np.ndarray:
        """
        combines the global scores and focus scores of images by weighted average
        :param global_scores:
        :param focus_scores:
        :param alpha: weight for average
        :return:
        """
        self.timer.start_measurement("TeranRetriever::compute_combined_scores")
        if alpha < 0. or alpha > 1.:
            raise ValueError("Alpha has to be between 0 and 1!")
        elif len(global_scores) != len(focus_scores):
            raise ValueError("There must be a global and a focus score for every image!")

        # first scale / normalize the scores with min max normalization so that both are in the same intervals from
        # 0 to 1.
        # otherwise, due to MrSw, the global_scores are always larger than the focus_scores
        # this is because MrSw sums the maxima of the rows for ALL tokens and the focus scores only contain the sums of
        # regions related to the focus!
        global_scores_normed = minmax_scale(global_scores.reshape(-1, 1)).squeeze()
        focus_scores_normed = minmax_scale(focus_scores.reshape(-1, 1)).squeeze()

        # combine the normalized scores
        comb_scores = np.array([alpha * foc_score + (1 - alpha) * glob_score
                                for foc_score, glob_score in zip(focus_scores_normed, global_scores_normed)])
        self.timer.stop_measurement()
        return comb_scores

    def compute_focus_scores(self,
                             focus: str,
                             context: str,
                             wra_matrices: np.ndarray,
                             focus_pooling: str = 'avg') -> np.ndarray:
        self.timer.start_measurement("TeranRetriever::compute_focus_scores")
        focus_span = self.find_focus_span_in_context(focus, context)
        focus_scores = np.array([self.compute_focus_score(wra, focus_span, focus_pooling) for wra in wra_matrices])
        self.timer.stop_measurement()
        return focus_scores

    def compute_focus_score(self,
                            wra_matrix: np.ndarray,
                            focus_span: Tuple[int, int],
                            focus_pooling: str = 'avg'):
        self.timer.start_measurement("TeranRetriever::compute_focus_score")
        if focus_pooling == 'avg':
            # avg of all focus WRAs
            focus_score = np.mean(wra_matrix[:, focus_span])
        elif focus_pooling == 'max':
            # max of the focus WRAs (max of all cols between start:end)
            focus_score = np.max(wra_matrix[:, focus_span])
        elif focus_pooling == 'max_avg':
            # maxima of the avg focus WRAs (max of avg of cols between start:end)
            focus_score = np.max(np.mean(wra_matrix[:, focus_span], axis=1))
        else:
            raise NotImplemented(f"Focus Pooling strategy {focus_pooling} is not implemented!")

        self.timer.stop_measurement()
        return focus_score

    def find_max_focus_region_index(self, focus_span: Tuple[int, int], wra_matrix: np.ndarray) -> int:
        max_focus_region_idx = np.argmax(np.mean(wra_matrix[:, focus_span], axis=1)).squeeze()
        logger.debug(f"Focus has strongest signal in region {max_focus_region_idx}!")
        return max_focus_region_idx

    def find_focus_span_in_context(self, focus: str, context: str) -> Tuple[int, int]:
        self.timer.start_measurement("TeranRetriever::find_focus_span_in_context")
        logger.debug(f"Searching focus span in context...")
        # TODO move removal of sep and cls token to config
        ctx_tokens, ctx_token_ids = self.tokenize(context, remove_sep_cls=True)
        focus_tokens, focus_token_ids = self.tokenize(focus, remove_sep_cls=True)

        begin_idx = ctx_token_ids.index(focus_token_ids[0])
        if len(focus_tokens) > 1:
            end_idx = ctx_token_ids.index(focus_token_ids[-1], begin_idx)
        else:
            end_idx = begin_idx

        logger.debug(f"Found focus span in context: {(begin_idx, end_idx)}!")
        self.timer.stop_measurement()

        return begin_idx, end_idx

    def tokenize(self, text: str, remove_sep_cls: bool = True) -> Tuple[List[str], List[int]]:
        self.timer.start_measurement("TeranRetriever::tokenize")
        token_ids = self.tokenizer.encode(text)
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        if remove_sep_cls:
            tokens = tokens[1:-1]
            token_ids = token_ids[1:-1]
        self.timer.stop_measurement()
        return tokens, token_ids

    @staticmethod
    def __build_retrieval_opts(device: str, model: str, model_config: str):
        opts = SimpleNamespace()
        opts.model = os.path.join(os.getcwd(), TERAN_PATH, model)
        opts.config = os.path.join(os.getcwd(), TERAN_PATH, model_config)
        opts.device = device
        return opts
