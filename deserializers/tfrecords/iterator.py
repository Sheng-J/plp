import tensorflow as tf


def simple_iterator(tfrecords_path, parse_f):
    dataset = tf.data.TFRecordDataset(tfrecords_path)
    dataset = dataset.map(parse_f)
    iterator = dataset.make_initializable_iterator()
    return iterator.initializer, iterator.get_next()


def batched_iterator(tfrecords_path, parse_f, batch_size):
    dataset = tf.data.TFRecordDataset(tfrecords_path)
    dataset = dataset.map(parse_f)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    return iterator.initializer, iterator.get_next()


def batched_drop_ramainder_iterator(tfrecords_path, parse_f, batch_size):
    dataset = tf.data.TFRecordDataset(tfrecords_path)
    dataset = dataset.map(parse_f)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    iterator = dataset.make_initializable_iterator()
    return iterator.initializer, iterator.get_next()


def batched_drop_remainder_dataset(tfrecords_path, parse_f, batch_size):
    dataset = tf.data.TFRecordDataset(tfrecords_path)
    dataset = dataset.map(parse_f)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    return dataset

