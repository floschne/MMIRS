import pickle
import time

import faiss
import numpy as np
from loguru import logger
from pathlib import Path
from sentence_transformers import util, SentenceTransformer
from typing import Dict, Any

from config import conf


def verify_embedding_structure(emb_struct: Dict[str, Any]) -> bool:
    for key in ['corpus_ids', 'embeddings', 'type', 'model', 'dataset']:
        assert key in emb_struct.keys(), f"Cannot find key '{key}' in Sentence Embedding Structure!"
    return True


def load_sentence_embeddings(path: Path) -> Dict[str, Any]:
    assert path.exists(), f"Cannot read {path}!"
    logger.info(f"Loading Sentence Embedding Structure from {str(path)}")
    with open(str(path), "rb") as fIn:
        data = pickle.load(fIn)
        verify_embedding_structure(data)
        logger.info(f"Successfully loaded {data['type']} Sentence Embeddings for {data['model']}!")
        return data


class ContextPreselector(object):
    __singleton = None

    def __new__(cls, *args, **kwargs):
        if cls.__singleton is None:
            logger.info('Instantiating Context Preselector!')
            cls.__singleton = super(ContextPreselector, cls).__new__(cls)

            pssc_conf = conf.preselection.context

            # setup sentence transformers (sbert)
            cls.symmetric_model = pssc_conf.sbert.symmetric_model
            cls.asymmetric_model = pssc_conf.sbert.asymmetric_model
            cls.max_seq_len = pssc_conf.sbert.max_seq_len

            if not pssc_conf.use_symmetric and not pssc_conf.use_asymmetric:
                logger.error("Both, use_symmetric and use_asymmetric are set to False!")
                SystemError("Both, use_symmetric and use_asymmetric are set to False!")

            logger.info("Loading SentenceEmbeddings into Memory...")
            if pssc_conf.use_symmetric:
                cls.symmetric_embeddings = {
                    k: load_sentence_embeddings(Path(v)) for d in pssc_conf.sbert.symmetric_embeddings for k, v in
                    d.items()
                }
            if pssc_conf.use_asymmetric:
                cls.asymmetric_embeddings = {
                    k: load_sentence_embeddings(Path(v)) for d in pssc_conf.sbert.asymmetric_embeddings for k, v in
                    d.items()
                }

            logger.info("Loading SentenceTransformer Models into Memory...")
            cls.sembedders = {}
            if pssc_conf.use_symmetric:
                cls.sembedders['symm'] = SentenceTransformer(cls.symmetric_model)
            if pssc_conf.use_asymmetric:
                cls.sembedders['asym'] = SentenceTransformer(cls.asymmetric_model)

            # setup faiss
            # TODO check comment regarding nprobe for FlatIPIndex quantizer on github
            cls.faiss_nprobe = pssc_conf.faiss.nprobe
            logger.info("Loading FAISS Indices into Memory...")
            if pssc_conf.use_symmetric:
                cls.symmetric_indices = {
                    k: faiss.read_index(v) for d in pssc_conf.faiss.symmetric_indices for k, v in d.items()
                }
            if pssc_conf.use_asymmetric:
                cls.asymmetric_indices = {
                    k: faiss.read_index(v) for d in pssc_conf.faiss.asymmetric_indices for k, v in d.items()
                }

        return cls.__singleton

    def faiss_index_available_for_dataset(self, dataset: str, symmetric: bool):
        indices = self.symmetric_indices if symmetric else self.asymmetric_indices
        return dataset in indices.keys()

    def sentence_embeddings_available_for_dataset(self, dataset: str, symmetric: bool):
        embs = self.symmetric_embeddings if symmetric else self.asymmetric_embeddings
        return dataset in embs.keys()

    def retrieve_top_k_relevant_images(self,
                                       context: str,
                                       k: int,
                                       dataset: str,
                                       exact: bool = False) -> Dict[str, float]:
        """
        Retrives the top-k relevant images by comparing the context with the captions of the specified dataset
        :param context: the context (of a RetrievalRequest) a sentence(s).
        :type context:
        :param k: specifies how many relevant images will be returned
        :type k:
        :param dataset: the context will be compared to the dataset specified by this parameter
        :type dataset:
        :param exact: if True, the context is compared to every caption in the dataset. If False an approximated search
        is done.
        :type exact:
        :return: a dictionary containing the top-k relevant images. Keys are image ids. Values are relevance scores.
        :rtype:
        """
        # TODO for now we only use symmetric
        #   in later versions we want to decide this dynamically by analysing the query (embedding)
        logger.debug(
            f"Retrieving top-{k} relevant images with exact={exact} in dataset {dataset} for context {context}")
        start = time.time()
        # compute context embedding
        context_embedding = self.sembedders['symm'].encode(context)

        # normalize vector to unit length, so that inner product is equal to cosine similarity
        context_embedding = context_embedding / np.linalg.norm(context_embedding)
        context_embedding = np.expand_dims(context_embedding, axis=0)
        if not exact:
            if not self.faiss_index_available_for_dataset(dataset, symmetric=True):
                logger.error(f"FAISS Index for dataset {dataset} not available!")
                raise FileNotFoundError(f"FAISS Index for dataset {dataset} not available!")

            index = self.symmetric_indices[dataset]

            # Approximate Nearest Neighbor (ANN) on FAISS INV Index (Voronoi Cells)
            # returns a matrix with distances and corpus ids.
            index.nprobe = self.faiss_nprobe
            distances, cids = index.search(context_embedding, k)

            # We extract corpus ids and scores for the first query
            hits = [{'corpus_id': cid, 'score': score} for cid, score in zip(cids[0], distances[0])]
        else:
            if not self.sentence_embeddings_available_for_dataset(dataset, symmetric=True):
                logger.error(f"Sentence Embeddings for dataset {dataset} not available!")
                raise FileNotFoundError(f"Sentence Embeddings for dataset {dataset} not available!")

            embs = self.symmetric_embeddings[dataset]['embeddings']

            # Approximate Nearest Neighbor (ANN) is not exact, it might miss entries with high cosine similarity / dot p
            # --> use exact search from sbert
            hits = util.semantic_search(context_embedding,
                                        embs,
                                        top_k=k)[0]

        # sort by score
        hits = sorted(hits, key=lambda x: x['score'], reverse=True)
        # look up the document ids of the hits (the hits contain indices but we need the document id)
        corpus_ids = self.symmetric_embeddings[dataset]['corpus_ids']
        top_k_matches = {str(corpus_ids[hit['corpus_id']]): hit['score'] for hit in hits[:k]}
        logger.info(f"Retrieving top-{k} relevant images with exact={exact} took {time.time() - start}s")
        # TODO add option to return the caption texts -> load the dataset dataframes and return the caps by id
        return top_k_matches
