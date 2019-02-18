import abc
import collections
import plp.token as ptoken


class DocTransformer(abc.ABC):

    @abc.abstractmethod
    def get_lists_stats(self, *docs):
        pass

    @abc.abstractmethod
    def transform_docs(self, *docs):
        pass

    @abc.abstractmethod
    def transform_seq_docs(self, *seq_docs):
        pass

    @abc.abstractmethod
    def estimate_docs_transformed_size(self, *docs):
        pass

    @abc.abstractmethod
    def estimate_seq_docs_transformed_size(self, *docs):
        pass

    @staticmethod
    def docs_consist_check(*docs):
        for doc in docs:
            assert doc.token_type == docs[0].token_type

    @staticmethod
    def assert_docs_token_type(token_type, *docs):
        for doc in docs:
            assert doc.token_type == token_type


class DocTransformState(collections.namedtuple(
    "DocTransformState",
    ("docs", "transformer", "size")
    )):
    pass


class ListStat:
    def __init__(self, name, token_type, list_len,
                 is_seq=False, sub_list_stat=None):
        self._name = name
        self._token_type = token_type
        self._list_len = list_len
        self._is_seq = is_seq
        if self._token_type != "list_type":
            assert sub_list_stat is None
        else:
            assert sub_list_stat is not None
        self._sub_list_stat = sub_list_stat

    @property
    def name(self):
        return self._name

    @property
    def token_type(self):
        return self._token_type

    @property
    def is_seq(self):
        return self._is_seq

    @property
    def list_len(self):
        return self._list_len

    @property
    def sub_list_stat(self):
        return self._sub_list_stat

