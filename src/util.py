# coding=utf-8
import numpy as np
import tensorflow as tf


def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), indices.max(0)[1] + 1], dtype=np.int64)

    # return tf.SparseTensor(indices=indices, values=values, shape=shape)
    return indices, values, shape


def sparsetensor_show(st):
    b = tf.sparse_tensor_to_dense(st, default_value=-1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res1, res2 = sess.run([st, b])
        print("res1:{}".format(res1))
        print("res2:{}".format(res2))


def get_sparsetensor(indices_, values_, dense_shape_):
    st = tf.SparseTensor(indices_, values_, dense_shape_)
    return st


if __name__ == '__main__':
    s = [[10, 27, 32, 14, 50, 6], [1, 2, 3, 4, 5, 9], [11, 22, 33, 44, 55, 666, 77, 88, 99, 400, 1111]]
    indices_t, values_t, shape_t = sparse_tuple_from(s)
    print(indices_t, values_t, shape_t)
    print([1] * 10)
    zipped = zip([0] * 10, [500, 2, 3, 4, 100])
    print(list(zipped))
    s_t = get_sparsetensor(indices_t, values_t, shape_t)
    sparsetensor_show(s_t)
    print(s)
