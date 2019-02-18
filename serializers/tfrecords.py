import tensorflow as tf
import plp.token as ptoken
from plp.utils.iterator import limit_iter
from functools import partial
import pdb


def _bytes_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[value.encode()]
        )
    )


def _bytes_features(values):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[value.encode() for value in values]
        )
    )


def _int64_feature(value):
    return tf.train.Feature(
        int64_list=tf.train.Int64List(
            value=[value]
        )
    )


def _int64_features(values):
    return tf.train.Feature(
        int64_list=tf.train.Int64List(
            value=values
        )
    )


def _float64_feature(value):
    return tf.train.Feature(
        float_list=tf.train.FloatList(
            value=[value]
        )
    )


def _float64_features(values):
    return tf.train.Feature(
        float_list=tf.train.FloatList(
            value=values
        )
    )


def _feature_list(value_list, feature_func):
    return tf.train.FeatureList(
        feature=[feature_func(v) for v in value_list]
    )


def _feature_dict(
    int_feature_dict, bytes_feature_dict, float_feature_dict,
    int_features_dict, bytes_features_dict, float_features_dict
):
    feature_dict = {}
    for key, val in int_feature_dict.items():
        feature_dict[key] = _int64_feature(val)
    for key, val in bytes_feature_dict.items():
        feature_dict[key] = _bytes_feature(val)
    for key, val in float_feature_dict.items():
        feature_dict[key] = _float64_feature(val)
    for key, vals in int_features_dict.items():
        feature_dict[key] = _int64_features(vals)
    for key, vals in bytes_features_dict.items():
        feature_dict[key] = _bytes_features(vals)
    for key, vals in float_features_dict.items():
        feature_dict[key] = _float64_features(vals)
    return feature_dict


def _feature_lists_dict(
    int_feature_list_dict, bytes_feature_list_dict, float_feature_list_dict,
    int_features_list_dict, bytes_features_list_dict, float_features_list_dict
):
    feature_dict = {}
    for key, val_list in int_feature_list_dict.items():
        feature_dict[key] = _feature_list(val_list, _int64_feature)
    for key, val_list in bytes_feature_list_dict.items():
        feature_dict[key] = _feature_list(val_list, _bytes_feature)
    for key, val_list in float_feature_list_dict.items():
        feature_dict[key] = _feature_list(val_list, _float64_feature)
    for key, vals_list in int_features_list_dict.items():
        feature_dict[key] = _feature_list(vals_list, _int64_features)
    for key, vals_list in bytes_features_list_dict.items():
        feature_dict[key] = _feature_list(vals_list, _bytes_features)
    for key, vals_list in float_features_list_dict.items():
        feature_dict[key] = _feature_list(vals_list, _float64_features)
    return feature_dict


def make_example(feature_dict):
    ex = tf.train.Example(
        features=tf.train.Features(
            features=feature_dict
        )
    )
    return ex


def make_sequence_example(context_feature_dict, feature_list_dict):
    ex = tf.train.SequenceExample(
        context=tf.train.Features(
            feature=context_feature_dict
        ),
        feature_lists=tf.train.FeatureLists(
            feature_list=feature_list_dict
        )
    )
    return ex


def docs_transformed_save(doc_transform_state, save_path):
    iterator = doc_transform_state.transformer.transform_docs(
        *doc_transform_state.docs
    )
    lists_stats = doc_transform_state.transformer.get_lists_stats(
        *doc_transform_state.docs
    )
    _transformed_save(iterator, lists_stats, save_path, doc_transform_state.size)


def seq_docs_transformed_save(doc_transform_state, save_path):
    iterator = doc_transform_state.transformer.transform_seq_docs(
        *doc_transform_state.docs
    )
    lists_stats = doc_transform_state.transformer.get_lists_stats(
        *doc_transform_state.docs
    )
    _transformed_save(iterator, lists_stats, save_path, doc_transform_state.size)


def _transformed_save(lists_iter, lists_stats, save_path, size):
    feature_fs = []
    context_type, feature_list_type = 0, 1
    for list_stat in lists_stats:
        token_type = list_stat.token_type
        if token_type == "list_type":
            sub_token_type = list_stat.sub_list_stat.token_type
            if sub_token_type == "list_type":
                raise ValueError("Not supported")
            else:
                if sub_token_type == "word_type":
                    feature_f = partial(_feature_list, feature_func=_bytes_features)
                elif sub_token_type == "id_type" or token_type == "value_int_type":
                    feature_f = partial(_feature_list, feature_func=_int64_features)
                elif sub_token_type == "value_float_type":
                    feature_f = partial(_feature_list, feature_func=_float64_features)
                else:
                    raise ValueError("not supported")
                feature_fs.append((feature_f, feature_list_type, list_stat.name))
        else:
            if list_stat.is_seq:
                if token_type == "word_type":
                    feature_f = partial(_feature_list, feature_func=_bytes_feature)
                elif token_type == "id_type" or token_type == "value_int_type":
                    feature_f = partial(_feature_list, feature_func=_int64_feature)
                elif token_type == "value_float_type":
                    feature_f = partial(_feature_list, feature_func=_float64_features)
                else:
                    raise ValueError("not supported")
                feature_fs.append((feature_f, feature_list_type, list_stat.name))
            else:
                if list_stat.list_len > 1:
                    if token_type == "word_type":
                        feature_fs.append((_bytes_features, context_type, list_stat.name))
                    elif token_type == "id_type" or token_type == "value_int_type":
                        feature_fs.append((_int64_features, context_type, list_stat.name))
                    elif token_type == "value_float_type":
                        feature_fs.append((_float64_features, context_type, list_stat.name))
                    else:
                        raise ValueError("not supported")
                else:
                    if token_type == "word_type":
                        feature_fs.append((_bytes_feature, context_type, list_stat.name))
                    elif token_type == "id_type" or token_type == "value_int_type":
                        feature_fs.append((_int64_feature, context_type, list_stat.name))
                    elif token_type == "value_float_type":
                        feature_fs.append((_float64_feature, context_type, list_stat.name))
                    else:
                        raise ValueError("not supported")

    with tf.python_io.TFRecordWriter(save_path) as writer:
        if size is not None:
            lists_iter = limit_iter(lists_iter, size)
        for lists in lists_iter:
            context_feature_dict, feature_lists_dict = {}, {}
            for t_list, feature_f_tuple in zip(lists, feature_fs):
                if feature_f_tuple[1] == context_type:
                    context_feature_dict[feature_f_tuple[2]] = feature_f_tuple[0](t_list)
                else:
                    feature_lists_dict[feature_f_tuple[2]] = feature_f_tuple[0](t_list)
            seq_ex = make_sequence_example(context_feature_dict, feature_lists_dict)
            writer.write(seq_ex.SerializeToString())
