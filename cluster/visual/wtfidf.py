from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from cluster.visual import ImageMetadata


def build_wtf_idf_index(docs: List[ImageMetadata], out_path: Path) -> None:
    # get all terms of all docs
    terms = []
    for doc in docs:
        terms.extend(doc.terms)
    terms = set(terms)

    # document frequency (how many docs contain t)
    df = {t: 0 for t in terms}
    for t in tqdm(terms, desc="Computing Document Frequencies..."):
        for d in docs:
            if t in d.terms:
                df[t] += 1

    # smoothed, idf
    # idf(t) = log(N/(df + 1))
    idf = {t: np.log(len(df) / (df[t] + 1)) for t in df.keys()}

    # weighted term-frequency
    wtf = {}
    for d in tqdm(docs, desc="Collecting weighted term-frequencies..."):
        for t, td in d.term_data.items():
            wtf[t, d.img_id] = td.wtf

    # weighted tf-idf
    # wtf-idf(t, d) = wtf(t, d) * idf(t)
    wtf_idf = {}
    for t_f in tqdm(wtf, desc="Computing weighted tf-idf..."):
        wtf_idf[t_f] = wtf[t_f] * idf[t_f[0]]
    # sort alphabetically by terms
    wtf_idf = {k: v for k, v in sorted(wtf_idf.items(), key=lambda i: i[0][0])}

    # create DataFrame
    rows = []
    for td, v in tqdm(wtf_idf.items(), desc="Building Index..."):
        rows.append((*td, v))
    df = pd.DataFrame(data=rows, columns=['term', 'doc', 'wtf_idf'])

    # persist
    p = out_path.joinpath('wtf_idf.index')
    df.to_feather(p)

    logger.info(f"Successfully persisted WTF-IDF index at {p}")


def retrieve_top_k_documents(query, wtf_idf_index: pd.DataFrame):
    q_tokens = query.split(' ')
    doc_scores = {}
    for qt in q_tokens:
        # TODO here we want use glove or w2v to find non-exact but similar matches
        #  --> persist the vectors for each term in the dataframe and compare it with the vector for the query token
        rows = wtf_idf_index[wtf_idf_index['term'] == qt]
        for idx, row in rows.iterrows():
            try:
                doc_scores[row['doc']] += row['wtf_idf']
            except:
                doc_scores[row['doc']] = row['wtf_idf']

    return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
