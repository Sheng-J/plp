import tensorflow as tf


def create_qa_parse_f(context_len):
    def qa_parse(example_proto):
        context, sequence = tf.parse_single_sequence_example(
            serialized=example_proto,
            context_features={
                "q_len": tf.FixedLenFeature([], dtype=tf.int64),
                "cs_size": tf.FixedLenFeature([], dtype=tf.int64),
                "a_len": tf.FixedLenFeature([], dtype=tf.int64),
            },
            sequence_features={
                "c_lens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
                "question": tf.FixedLenSequenceFeature([], dtype=tf.string),
                "answer": tf.FixedLenFeature([], dtype=tf.string),
                "contexts": tf.FixedLenFeature([context_len], dtype=tf.string),
            }
        )
        return sequence["question"], sequence["contexts"], sequence["answer"], \
           context["q_len"], context["cs_size"], \
           context["a_len"], sequence["c_lens"]
    return qa_parse
