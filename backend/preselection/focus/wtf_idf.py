import os
import re

import pandas as pd
from loguru import logger


class WTFIDF(object):
    def __init__(self, file: str, dataset: str, doc_id_prefix: str):
        if not os.path.lexists(file) or not os.path.isfile(file):
            logger.error(f"Cannot read WTF-IDF Index for dataset {dataset} at {file}!")
            raise FileNotFoundError(f"Cannot read WTF-IDF Index for dataset {dataset} at {file}!")
        logger.info(f"Loading WTF-IDF Index for dataset {dataset}...")

        self.file = file
        self.dataset = dataset
        self.doc_id_prefix = doc_id_prefix

        # load the index
        df = pd.read_feather(file)

        # remove doc_id_prefix
        df['doc'] = df['doc'].apply(lambda x: re.sub(doc_id_prefix, '', x))

        # set multi index
        mi = pd.MultiIndex.from_frame(df[['term', 'doc']])
        self.df = df[['wtf_idf']].set_index(mi)

        logger.info(f"Loaded WTF-IDF Index for {dataset} with {len(self.df)} entries!")

    def __len__(self):
        return len(self.df)
