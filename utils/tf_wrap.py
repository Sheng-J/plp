import tensorflow as tf
import plp.vocab as pvocab
from tensorflow.python.ops import lookup_ops

class VocabReader(object):
    def __init__(self, vocab_f_path, count_f_path=None):
        self._vocab_f_path = vocab_f_path
        self._count_f_path = count_f_path
        self._vocab_size = None
        self._vocab_counts = None
        with tf.variable_scope("vocab_lookup"):
            self._id2word_table = lookup_ops.index_to_string_table_from_file(
                self._vocab_f_path, default_value=pvocab.UNK, name="id2word"
            )
            self._word2id_table = lookup_ops.index_table_from_file(
                self._vocab_f_path, default_value=pvocab.UNK_ID, name="word2id"
            )

    def id2word(self, id_token):
        return self._id2word_table.lookup(tf.to_int64(id_token))

    def word2id(self, word_token):
        return tf.cast(self._word2id_table.lookup(word_token), tf.int32)

    @property
    def vocab_size(self):
        if not self._vocab_size:
            with open(self._vocab_f_path) as f:
                self._vocab_size = len(f.read().strip().split())
        return self._vocab_size

    @property
    def vocab_counts_list(self):
        if not self._vocab_counts and self._count_f_path:
            with open(self._count_f_path) as count_f:
                counts = count_f.read().strip().split()
                self._vocab_counts = [int(count) for count in counts]
        return self._vocab_counts

