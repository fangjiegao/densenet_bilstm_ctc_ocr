"""
    create model for train and predict
    illool@163.com
"""
import tensorflow as tf
import nets_factory
import Bilstm
import generator_tool


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    num_channels = images.get_shape().as_list()[-1]  # 获得多少个通道
    if len(means) != num_channels:  # 通道数是不是匹配
        raise ValueError('len(means) must match the number of channels')
    # 将axis=3,第四维num_channels分成num_channels份,每份shape=[images[0],images[1],images[2],1]
    # 3通道就是分成3份，每份代表一个通道，下面每个通道进行归一化
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)  # 切片:则value沿维度axis分割成为num_split
    for i in range(num_channels):
        channels[i] -= means[i]  # 遍历每个通道,归一化
    return tf.concat(axis=3, values=channels)  # 再合并成三通道


def create_loss(logits, labels, num_labels=1000):
    with tf.variable_scope("class_loss"):
        probabilities = tf.nn.softmax(logits, axis=-1)  # 对num_labels softmax [1, 128, num_labels]
        log_probs = tf.nn.log_softmax(logits, axis=-1)  # 对num_labels log_softmax [64, 128, num_labels]
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)  # 对应元素相乘
        loss = tf.reduce_mean(per_example_loss)
        predict = tf.argmax(probabilities, axis=-1)

    return loss, per_example_loss, logits, probabilities, predict


def create_model_fn(image, labels_sparse, sequence_lengths_t, n_classes=1000, batch_size=1):
    # image = mean_image_subtraction(image)  # 图像归一化,所有图片都是单通道的

    # image = tf.convert_to_tensor(image, image.dtype)
    # labels_sparse = tf.convert_to_tensor(labels_sparse, labels_sparse.dtype)
    # sequence_lengths_t = tf.convert_to_tensor(sequence_lengths_t, sequence_lengths_t.dtype)

    network_fn = nets_factory.get_network_fn(
        "densenet169_fine_tuning",
        num_classes=1000,
        data_format="NHWC",
        is_training=False)
    net = network_fn(image)
    print("net:", type(net), net.get_shape())
    net = tf.transpose(net, [0, 2, 1, 3])  # (1, 13, 19, 1664)=>(1, 19, 13, 1664)
    print("net:", type(net), net.get_shape(), net.get_shape()[1], net.get_shape()[2], net.get_shape()[3])
    # dim = tf.reduce_prod(tf.shape(net)[2:])  # dim = 13*1664
    shape_dim = net.get_shape().as_list()
    dim = shape_dim[2] * shape_dim[3]
    print("dim:", batch_size, shape_dim[1], dim)
    # print(tf.shape(net)[0], tf.shape(net)[1], dim)
    net = tf.reshape(net, [batch_size, -1, dim])  # shape_dim[1]
    # seq_len = tf.fill([shape_dim[0]], shape_dim[1])  # 生成sequence_lengths, shape_dim[1]=None不可用此方法
    seq_len = tf.divide(sequence_lengths_t, 32)
    # seq_len = tf.div(sequence_lengths_t, 32)
    seq_len = tf.cast(seq_len, dtype=tf.int32)
    # print("seq_featrue_flatten:", net.shape)  # tensor_shape.Dimension

    logits = Bilstm.bi_lstm_fc(net, batch_size=batch_size, n_classes=n_classes)  # [batch_size, time_steps, n_classes]
    print("logits:", logits.get_shape())

    # logits = tf.reshape(logits, [batch_size, -1, n_classes])  # [64, 768, num_labels], 不能写None而是-1
    # print("logits:", logits.get_shape())
    # time_major=False,[batch_size,max_time_step,num_classes];
    # logits = tf.reshape(logits, [max_time_step,batch_size,num_classes]) ==> time_major=True
    # print("sequence_lengths_t:", sequence_lengths_t.get_shape(), type(sequence_lengths_t))
    print(labels_sparse)
    targets_lable = generator_tool.dense2sparse(labels_sparse)
    print(seq_len)
    loss = tf.nn.ctc_loss(labels=targets_lable, inputs=logits, sequence_length=seq_len,
                          time_major=False, ignore_longer_outputs_than_inputs=True)
    print("loss:", loss)
    cost = tf.reduce_mean(loss)
    print("cost", cost)

    logits = tf.reshape(logits, [-1, batch_size, n_classes])
    print("logits:", logits.get_shape())
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(inputs=logits, sequence_length=seq_len,
                                                      merge_repeated=False)

    return cost, decoded


def create_model_predict_fn(image, sequence_lengths_t, n_classes=1000, batch_size=1):
    network_fn = nets_factory.get_network_fn(
        "densenet169_fine_tuning",
        num_classes=1000,
        data_format="NHWC",
        is_training=False)
    net = network_fn(image)
    print("net:", type(net), net.get_shape())
    net = tf.transpose(net, [0, 2, 1, 3])  # (1, 13, 19, 1664)=>(1, 19, 13, 1664)
    print("net:", type(net), net.get_shape(), net.get_shape()[1], net.get_shape()[2], net.get_shape()[3])
    # dim = tf.reduce_prod(tf.shape(net)[2:])  # dim = 13*1664
    shape_dim = net.get_shape().as_list()
    dim = shape_dim[2] * shape_dim[3]
    print("dim:", batch_size, shape_dim[1], dim)
    # print(tf.shape(net)[0], tf.shape(net)[1], dim)
    net = tf.reshape(net, [batch_size, -1, dim])  # shape_dim[1]
    print("net:", type(net), net.get_shape(), net.get_shape()[0], net.get_shape()[1], net.get_shape()[2])
    # seq_len = tf.fill([shape_dim[0]], shape_dim[1])  # 生成sequence_lengths, shape_dim[1]=None不可用此方法
    seq_len = tf.divide(sequence_lengths_t, 32)
    # seq_len = tf.div(sequence_lengths_t, 32)
    seq_len = tf.cast(seq_len, dtype=tf.int32)
    print("seq_len:", seq_len.shape, seq_len)

    logits = Bilstm.bi_lstm_fc(net, batch_size=batch_size, n_classes=n_classes)  # [batch_size, time_steps, n_classes]
    print("logits:", logits.get_shape())
    logits = tf.reshape(logits, [-1, batch_size, n_classes])
    print("logits:", logits.get_shape())
    # logits = tf.reshape(logits, [batch_size, -1, n_classes])  # [64, 768, num_labels], 不能写None而是-1
    # print("logits:", logits.get_shape())
    # time_major=False,[batch_size,max_time_step,num_classes];
    # logits = tf.reshape(logits, [max_time_step,batch_size,num_classes]) ==> time_major=True

    decoded, log_prob = tf.nn.ctc_beam_search_decoder(inputs=logits, sequence_length=seq_len,
                                                      merge_repeated=False)

    return decoded, log_prob
