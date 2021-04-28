import sys
from typing import List

from backend.fineselection.data import ImageFeaturePool, TeranISS
from backend.fineselection.retriever.retriever import RetrieverType

TERAN_PATH = 'models/teran'
sys.path.append(TERAN_PATH)
# noinspection PyUnresolvedReferences
from data import PreComputedImageEmbeddingsData


class TeranImageFeaturePool(ImageFeaturePool):
    def __init__(self,
                 source_dataset: str,
                 data_root: str,
                 fn_prefix: str,
                 pre_fetch: bool = False,
                 num_workers: int = 8):
        """
        :param source_dataset: The dataset the image features originate from
        :param pre_fetch: if True load the !complete! feature pool into memory
        :param data_root: the root directory where the features are located
        :param num_workers: The number of workers to load the features in parallel
        """
        super().__init__(source_dataset=source_dataset,
                         target_retriever_type=RetrieverType.TERAN,
                         pre_fetch=pre_fetch,
                         data_root=data_root)
        self.data = PreComputedImageEmbeddingsData(pre_computed_img_embeddings_root=data_root,
                                                   pre_fetch_in_memory=False,
                                                   fn_prefix=fn_prefix,
                                                   num_pre_fetch_workers=num_workers)

    def load_data_into_memory(self):
        self.data.fetch_img_embs()

    def get_image_search_space(self, img_ids: List[str]) -> TeranISS:
        if self.source_dataset == 'coco':
            # TODO fix this elsewhere (preferably in the Sentence Embedding Structure.
            #  There the leading 0 get removed because the id's are stored as integers)
            img_ids = [TeranImageFeaturePool.fill_leading_coco_zeros(img_id) for img_id in img_ids]
            
        subset = self.data.get_subset(image_ids=img_ids, pre_fetch_in_memory=True)
        return TeranISS(images=subset)

    @staticmethod
    def fill_leading_coco_zeros(img_id: str):
        while len(img_id) != 6:
            img_id = '0' + img_id
        return img_id
