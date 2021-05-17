from loguru import logger

from config import conf


class VisualVocab(object):
    __singleton = None

    def __new__(cls, *args, **kwargs):
        if cls.__singleton is None:
            logger.info(f"Instantiating Visual Vocabulary ...")
            cls.__singleton = super(VisualVocab, cls).__new__(cls)

            vocab_conf = conf.preselection.focus.vocab
            # Load object categories
            cls.objs_vocab = []
            with open(vocab_conf.objs_file) as f:
                for obj in f.readlines():
                    cls.objs_vocab.append(obj.split(',')[0].lower().strip())

            # Load attributes
            cls.attrs_vocab = []
            with open(vocab_conf.attrs_file) as f:
                for att in f.readlines():
                    cls.attrs_vocab.append(att.split(',')[0].lower().strip())
            logger.info(
                f"Loaded {len(cls.objs_vocab)} Object Categories and {len(cls.attrs_vocab)} Attribute Categories")

            # for some strange reasons there are terms in both lists. e.g. 'windshield' and 'wii'..
            cls.full_vocab = list(set(cls.objs_vocab + cls.attrs_vocab))

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

    def __len__(self):
        return len(self.full_vocab)

    def __contains__(self, term):
        return term in self.full_vocab

    def __getitem__(self, idx):
        return self.full_vocab[idx]
