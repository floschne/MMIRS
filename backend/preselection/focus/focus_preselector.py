import time

import numba
import numpy as np
import spacy
from loguru import logger
from pymagnitude import Magnitude
from typing import Dict, List, Optional, Union, Tuple

from backend.preselection import VisualVocab
from backend.preselection.focus.wtf_idf import WTFIDF
from config import conf


@numba.jit
def weight(row):
    return row.wtf_idf * row.weight


class FocusPreselector(object):
    __singleton = None

    def __new__(cls, *args, **kwargs):
        if cls.__singleton is None:
            logger.info('Instantiating Focus Preselector!')
            cls.__singleton = super(FocusPreselector, cls).__new__(cls)

            cls.vocab = VisualVocab()
            pssf_conf = conf.preselection.focus

            # load magnitude
            logger.info(f"Loading Magnitude Embeddings {pssf_conf.magnitude.embeddings}!")
            cls.magnitude = Magnitude(pssf_conf.magnitude.embeddings)
            cls.top_k_similar = pssf_conf.magnitude.top_k_similar
            cls.max_similar = pssf_conf.magnitude.max_similar

            # load wtf-idf indices
            logger.info(f"Loading WTF-IDF Indices!")
            cls.wtf_idf = {}
            for ds in pssf_conf.wtf_idf.keys():
                cls.wtf_idf[ds] = WTFIDF(file=pssf_conf.wtf_idf[ds].file,
                                         dataset=ds,
                                         doc_id_prefix=pssf_conf.wtf_idf[ds].doc_id_prefix)

            # setup spacy
            # TODO can we disable some pipeline components to be faster? e.g. ner
            logger.info(f"Loading spaCy model {pssf_conf.spacy_model}!")
            cls.spacy_nlp = spacy.load(pssf_conf.spacy_model)

            cls.focus_max_tokens = pssf_conf.max_tokens
            cls.focus_remove_stopwords = pssf_conf.remove_stopwords
            cls.focus_uncased = pssf_conf.uncased
            cls.focus_lemmatize = pssf_conf.lemmatize
            cls.focus_pos_tags = pssf_conf.pos_tags

            # perform warm up (first time takes about 20s)
            logger.info(f"Performing warmup...")
            cls.find_top_k_similar_focus_terms(cls.__singleton, "warmup")

        return cls.__singleton

    def pre_process_focus(self, focus: str) -> List[str]:
        logger.debug(f"Preprocessing focus term: {focus}")
        focus_terms = []
        for token in self.spacy_nlp(focus):
            if self.focus_remove_stopwords and token.is_stop:
                continue
            if not token.is_punct and token.pos_ in self.focus_pos_tags:
                # use lemma if lemmatize
                if self.focus_lemmatize:
                    tok = token.lemma_
                else:
                    tok = token.text
                # lower case if uncased
                if self.focus_uncased:
                    tok = tok.lower()

                focus_terms.append(tok)

        # naive fall back if spacy removed too many tokens
        if len(focus_terms) == 0:
            focus_terms = focus.split(' ')
            # add the full focus term
            focus_terms.append(focus)

        focus_terms = focus_terms[:self.focus_max_tokens]
        logger.debug(f"Preprocessed focus terms: {focus_terms}")
        return focus_terms

    @logger.catch
    def find_top_k_similar_focus_terms(self,
                                       focus: str,
                                       top_k_similar: Optional[int] = None,
                                       max_similar: Optional[int] = None) -> Dict[str, float]:
        # overwrite if set manually
        top_k_similar = self.top_k_similar if top_k_similar is None else top_k_similar
        max_similar = self.max_similar if max_similar is None else max_similar

        logger.debug(f"Finding top-{top_k_similar} similar focus terms for '{focus}'")
        focus_terms = self.pre_process_focus(focus)

        # term -> similarity as weight
        # largest weight for the 'original' terms if the terms are in the vocab
        similar_terms = {ft: 1. for ft in focus_terms if ft in self.vocab}

        # compute the similarities between each focus term and all terms in the vocab
        # then keep the top-k and add the term-similarity pairs to the similar terms mapping
        for ft in focus_terms:
            sims = self.magnitude.similarity(ft, self.vocab.full_vocab)
            similar_terms.update({self.vocab[idx]: sim for idx, sim in zip(np.argsort(sims)[-top_k_similar:][::-1],
                                                                           sorted(sims)[-top_k_similar:][::-1])})
        # sort by similarity and keep only max_similar
        similar_terms = {k: min(v, 1.) for k, v in
                         sorted(similar_terms.items(), key=lambda i: i[1], reverse=True)[:max_similar]}

        logger.debug(f"Found {len(similar_terms)} similar focus terms: {similar_terms}")
        return similar_terms

    @logger.catch
    def retrieve_top_k_relevant_images(self, focus: str,
                                       dataset: str,
                                       k: int = 100,
                                       weight_by_sim: bool = False,
                                       top_k_similar: Optional[int] = None,
                                       max_similar: Optional[int] = None,
                                       return_similar_terms: Optional[bool] = False) -> \
            Union[Tuple[Dict[str, float], List[str]], Dict[str, float]]:
        logger.debug(f"Retrieving top-{k} relevant images in dataset {dataset} for focus term {focus}")
        if dataset not in self.wtf_idf:
            logger.error(f"WTF-IDF Index for dataset {dataset} not available!")
            raise FileNotFoundError(f"WTF-IDF Index for dataset {dataset} not available!")

        wtf_idf = self.wtf_idf[dataset]

        start = time.time()
        similar_terms = self.find_top_k_similar_focus_terms(focus=focus,
                                                            top_k_similar=top_k_similar,
                                                            max_similar=max_similar)
        logger.debug(f"get_top_k_similar_terms took {time.time() - start}s")

        # remove terms which are not in the index
        start = time.time()
        # TODO make this more efficient
        similar_terms = {t: s for t, s in similar_terms.items() if t in wtf_idf.df.index}
        logger.debug(f"remove terms which are not in the index took {time.time() - start}s")

        # get the relevant entries (i.e. entries that match the similar terms)
        start = time.time()
        entries = wtf_idf.df.loc[similar_terms.keys()]
        logger.debug(f"get the relevant entries took {time.time() - start}s")

        if weight_by_sim:
            start = time.time()
            # TODO how to boost performance of creating the weights dict?!
            weights = np.array([similar_terms[t] for t, _ in entries.index])
            # create temporary weight dataframe
            # weights = pd.DataFrame(data=[*similar_terms.items()], columns=['term', 'weight']).set_index('term')

            # weight the wtf-idf scores by multiplying with the similarity of the term
            # entries = entries.assign(**weights).drop(columns=['weight'])
            entries['wtf_idf'] = entries['wtf_idf'].to_numpy() * weights
            # entries = entries.assign(**weights, wtf_idf=lambda x: x.wtf_idf * x.weight).drop(columns=['weight'])
            # entries = entries.assign(**weights, wtf_idf=weight).drop(columns=['weight'])
            logger.debug(f"weight_by_sim took {time.time() - start}s")

        # sum the wtf-idf scores and sort descending
        start = time.time()
        entries = entries.groupby('doc').sum()
        logger.debug(f"sum the wtf-idf scores took {time.time() - start}s")

        start = time.time()
        entries = entries.sort_values(by='wtf_idf', ascending=False)
        logger.debug(f"sort descending took {time.time() - start}s")

        # return dict
        entries_dict: Dict[str, float] = entries[:k].to_dict()['wtf_idf']
        if return_similar_terms:
            return entries_dict, list(similar_terms.keys())
        else:
            return entries_dict

    # naive implementation way to slow (86s for a three token focus word)
    # @logger.catch(reraise=True)
    # def retrieve_top_k_relevant_images(self, focus, k: int = 100):
    #     similar_terms = self.get_top_k_similar_terms(focus)
    #     doc_scores = {}
    #
    #     for term, sim in similar_terms.items():
    #         if term in self.wtf_idf.index:
    #             # get the relevant rows
    #             rows = self.wtf_idf.xs(term)
    #             for _, row in rows.iterrows():
    #                 # accumulate the scores for the respective documents
    #                 #   -> weight the wtf-idf scores by the term similarity
    #                 #   -> the more similar the term, the more relevant is the doc in which the term occurs
    #                 try:
    #                     doc_scores[row['doc']] += sim * row['wtf_idf']
    #                 except:
    #                     doc_scores[row['doc']] = sim * row['wtf_idf']
    #
    #     # return descending sorted docs
    #     return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]
