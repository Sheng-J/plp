import tensorflow as tf


def create_word2vec_parse_f():
    def word2vec_parse(example_proto):
        context, sequence = tf.parse_single_sequence_example(
            serialized=example_proto,
            context_features={
                "center": tf.FixedLenFeature([], dtype=tf.int64),
                "context": tf.FixedLenFeature([], dtype=tf.int64)
            },
            sequence_features={}
        )
        return context["center"], context["context"]
    return word2vec_parse
