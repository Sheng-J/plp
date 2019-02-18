import tensorflow as tf


def create_doc_classify_parse_f():
    def doc_classify_parse(example_proto):
        context, sequence = tf.parse_single_sequence_example(
                serialized=example_proto,
                context_features={
                    "src_len": tf.FixedLenFeature([], dtype=tf.int64),
                    "label": tf.FixedLenFeature([], dtype=tf.int64),
                    "eod_flag": tf.FixedLenFeature([], dtype=tf.int64)
                    },
                sequence_features={
                    "src": tf.FixedLenSequenceFeature([], dtype=tf.int64)
                    }
                )
        return sequence["src"], context["src_len"], context["label"], context["eod_flag"]
    return doc_classify_parse

def create_doc_classify_parse_old_f():
    def _parse(example_proto):
        context, sequence = tf.parse_single_sequence_example(
            serialized=example_proto,
            context_features={
                "label": tf.FixedLenFeature([], dtype=tf.int64),
                "eod_flag": tf.FixedLenFeature([], dtype=tf.int64)
            },
            sequence_features={
                "src": tf.FixedLenSequenceFeature([], dtype=tf.int64)
                }
            )
        return sequence["src"], context["label"], context["eod_flag"]
    return _parse

