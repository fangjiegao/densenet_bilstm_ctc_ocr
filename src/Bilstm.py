"""
    Bidirectional RNN
    illool@163.com
"""
import tensorflow as tf
state_size = 512  # hidden layer num of features


# feature_map:(1, 7, 11648)
def bi_lstm_fc(feature_map, batch_size=1, n_classes=1000):
    time_steps = tf.shape(feature_map)[1]
    # 双向rnn
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(state_size)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(state_size)
    print("feature_map shape", time_steps, batch_size, tf.shape(feature_map))
    init_fw = lstm_fw_cell.zero_state(batch_size, dtype=tf.float32)
    init_bw = lstm_bw_cell.zero_state(batch_size, dtype=tf.float32)

    # for fc
    weights = tf.get_variable("weights", [2 * state_size, n_classes], dtype=tf.float32,  # 注意这里的维度
                              initializer=tf.random_normal_initializer(mean=0, stddev=1))
    biases = tf.get_variable("biases", [n_classes], dtype=tf.float32,
                             initializer=tf.random_normal_initializer(mean=0, stddev=1))

    outputs, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                            lstm_bw_cell,
                                                            inputs=feature_map,
                                                            dtype=tf.float32,
                                                            initial_state_fw=init_fw,
                                                            initial_state_bw=init_bw
                                                            )

    outputs = tf.concat(outputs, 2)  # 将前向和后向的状态连接起来
    state_out = tf.matmul(tf.reshape(outputs, [-1, 2 * state_size]), weights) + biases  # 注意这里的维度
    logits = tf.reshape(state_out, [batch_size, time_steps, n_classes])
    return logits
