import os
import tensorflow as tf
import create_model
import gen_and_get_lable_set
import reshape_data_224
import numpy as np

slim = tf.contrib.slim
PROJECT_PATH = os.path.dirname(os.path.abspath(os.getcwd()))

# tf.app.flags.DEFINE_string('checkpoint_path', PROJECT_PATH + os.sep + "ckpt", '')
tf.app.flags.DEFINE_string('checkpoint_path', "/Users/sherry/work/pycharm_python/dense_net_restore" + os.sep + "ckpt", '')
tf.app.flags.DEFINE_string('logs_path', PROJECT_PATH + os.sep + "log", '')
tf.app.flags.DEFINE_boolean('restore', False, '')  # init:False, orther:Ture
tf.app.flags.DEFINE_float('learning_rate', 1e-5, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_integer('training_epochs', 100, '')
tf.app.flags.DEFINE_float('decay_rate', 0.1, '')
tf.app.flags.DEFINE_integer('decay_steps', 30000, '')
tf.app.flags.DEFINE_integer('max_to_keep', 10, '')

FLAGS = tf.app.flags.FLAGS

lable_set = gen_and_get_lable_set.read_lable_dict()
NUM_lable = len(lable_set.keys())
lable_map = gen_and_get_lable_set.read_lable_dict_opposite()
BATCH_SIZE = 1


def main(argv=None):
    file_path = r"/Users/sherry/data/Synthetic_Chinese_String_Dataset_part/image/20436796_3166386750.jpg"

    if not os.path.exists(FLAGS.checkpoint_path):
        os.makedirs(FLAGS.checkpoint_path)

    input_image = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 224, None, 3], name='input_image')
    # input_image_shape = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3], name='input_image_shape')  # [[HWC],[]...]
    # targets = tf.sparse_placeholder(tf.int32)
    seq_len = tf.placeholder(tf.int32, [BATCH_SIZE])

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    decoded, log_prob = create_model.create_model_predict_fn(
        input_image, sequence_lengths_t=seq_len, n_classes=NUM_lable, batch_size=BATCH_SIZE)

    # 滑动平均模型的作用是提高测试值上的健壮性, 模型参数进行平均得到的模型往往比单个模型的结果要好很多
    # decay = min(decay, (1 + num_updates) / (10 + num_updates))
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())

    with tf.Session() as sess:
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        saver.restore(sess, ckpt)

        img, img_w = reshape_data_224.reshape_picture_224(file_path)
        # img = img * (1. / 255) - 0.5
        img = img * (1. / 255)
        img = np.expand_dims(img, axis=0)
        img_w = np.expand_dims(img_w, axis=0)
        print(type(img), img.shape)
        print(type(img_w), img_w.shape, img_w)
        # feed 数据
        decoded, log_prob = sess.run([decoded, log_prob], feed_dict={input_image: img, seq_len: img_w})
        print("decoded:", decoded)
        decoded_dense = tf.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values,
                                           default_value=-1)
        print("decoded_dense:", type(decoded_dense.eval()), decoded_dense)
        label_predicts_idx = [_ for _ in decoded_dense.eval()[0]]
        label_predicts = [lable_map[_] for _ in label_predicts_idx]
        print(''.join(label_predicts))
        print(label_predicts_idx)


if __name__ == '__main__':
    tf.app.run()
