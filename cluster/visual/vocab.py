from loguru import logger

from backend.util import SingletonABCMeta


class Vocab(object):
    __metaclass__ = SingletonABCMeta

    def __init__(self,
                 name: str = "default",
                 obj_vocab_path: str = 'objects_vocab.txt',
                 attr_vocab_path: str = 'attributes_vocab.txt'):
        self.name = name
        logger.info(f"Instantiating {self.name} Visual Vocabulary ...")

        # Load object categories
        self.objs_vocab = []
        with open(obj_vocab_path) as f:
            for obj in f.readlines():
                self.objs_vocab.append(obj.split(',')[0].lower().strip())

        # Load attributes
        self.attrs_vocab = []
        with open(attr_vocab_path) as f:
            for att in f.readlines():
                self.attrs_vocab.append(att.split(',')[0].lower().strip())
        logger.info(f"Loaded {self.num_objs} Object Categories and {self.num_attrs} Attribute Categories")

    def get_obj_name(self, obj_id: int):
        return self.objs_vocab[obj_id]

    def get_attr_name(self, obj_id: int):
        return self.objs_vocab[obj_id]

    @property
    def num_objs(self):
        return len(self.objs_vocab)

    @property
    def num_attrs(self):
        return len(self.attrs_vocab)

    def __len__(self):
        return self.num_objs
