from loguru import logger
from omegaconf import OmegaConf


class VisualVocab(object):
    __singleton = None

    def __new__(cls, *args, **kwargs):
        if cls.__singleton is None:
            logger.info(f"Instantiating Visual Vocabulary ...")
            cls.__singleton = super(VisualVocab, cls).__new__(cls)

            conf = OmegaConf.load('config.yaml').preselection.focus.vocab
            # Load object categories
            cls.objs_vocab = []
            with open(conf.objs_file) as f:
                for obj in f.readlines():
                    cls.objs_vocab.append(obj.split(',')[0].lower().strip())

            # Load attributes
            cls.attrs_vocab = []
            with open(conf.attrs_file) as f:
                for att in f.readlines():
                    cls.attrs_vocab.append(att.split(',')[0].lower().strip())
            logger.info(
                f"Loaded {len(cls.objs_vocab)} Object Categories and {len(cls.attrs_vocab)} Attribute Categories")

        return cls.__singleton

    def get_obj_name(self, obj_id: int):
        return self.objs_vocab[obj_id]

    def get_attr_name(self, obj_id: int):
        return self.attrs_vocab[obj_id]

    @property
    def num_objs(self):
        return len(self.objs_vocab)

    @property
    def num_attrs(self):
        return len(self.attrs_vocab)

    @property
    def full_vocab(self):
        return self.objs_vocab + self.attrs_vocab

    def __len__(self):
        return self.num_objs + self.num_attrs

    def __contains__(self, term):
        return term in self.objs_vocab or term in self.attrs_vocab

    def __getitem__(self, idx):
        if idx < len(self.objs_vocab):
            return self.objs_vocab[idx]
        else:
            return self.attrs_vocab[idx - len(self.objs_vocab)]
