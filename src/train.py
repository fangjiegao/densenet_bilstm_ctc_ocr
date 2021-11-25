import os
import tensorflow as tf
import create_model
import datetime
import gen_and_get_lable_set
import generator_tool

slim = tf.contrib.slim
# PROJECT_PATH = os.path.dirname(os.path.abspath(os.getcwd()))
PROJECT_PATH = r"/home/gaofangjie"
# 预训练模型位置
tf.app.flags.DEFINE_string(
    # 'pretrained_model_path', '/Users/sherry/models/tf-densenet169/tf-densenet169.ckpt', 'model path')
    'pretrained_model_path', '/sys_a/tf-densenet169/tf-densenet169.ckpt', 'model path')
    # 'pretrained_model_path', None, 'model path')
tf.app.flags.DEFINE_string('checkpoint_path', PROJECT_PATH + os.sep + "ckpt", '')
tf.app.flags.DEFINE_string('logs_path', PROJECT_PATH + os.sep + "log", '')
tf.app.flags.DEFINE_boolean('restore', False, '')  # init:False, orther:True
tf.app.flags.DEFINE_float('learning_rate', 1e-5, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_integer('training_epochs', 100, '')
tf.app.flags.DEFINE_float('decay_rate', 0.1, '')
tf.app.flags.DEFINE_integer('decay_steps', 30000, '')
tf.app.flags.DEFINE_integer('max_to_keep', 5, '')
tf.app.flags.DEFINE_string(
    # 'tfrecord_path', '/Users/sherry/work/pycharm_python/generator_tfrecord_data/gscsd_train.tfrecord', 'data path')
    'tfrecord_path', '/home/gaofangjie/data', 'data path')
FLAGS = tf.app.flags.FLAGS

lable_set = gen_and_get_lable_set.read_lable_dict()
NUM_lable = len(lable_set.keys())
lable_map = gen_and_get_lable_set.read_lable_dict_opposite()
BATCH_SIZE = 1


def main(argv=None):
    # os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    now = datetime.datetime.now()
    styletime = now.strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(FLAGS.logs_path + styletime)
    if not os.path.exists(FLAGS.checkpoint_path):
        os.makedirs(FLAGS.checkpoint_path)

    input_image = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 224, None, 3], name='input_image')
    # input_image_shape = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3], name='input_image_shape')  # [[HWC],[]...]
    # targets = tf.sparse_placeholder(tf.int32)
    targets = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None], name='lable_seq')
    seq_len = tf.placeholder(tf.int32, [BATCH_SIZE])

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False)
    tf.summary.scalar('learning_rate', learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate)  # 优化器

    with tf.name_scope('model_ops') as scope:
        # total_loss, per_example_loss, logits, probabilities, predict = create_model.create_model_fn(
        total_loss, decoded_test = create_model.create_model_fn(
            input_image, sequence_lengths_t=seq_len, n_classes=NUM_lable, labels_sparse=targets, batch_size=BATCH_SIZE)
        # 这是一个tensorflow的计算图中内置的一个集合，其中会保存一些需要在训练操作之前完成的操作，并配合tf.control_dependencies函数使用
        batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
        # 简单说该函数就是用于计算loss对于指定val_list的导数的，最终返回的是元组列表，即[(gradient, variable),...]
        grads = opt.compute_gradients(total_loss)
    # 该函数的作用是将compute_gradients()返回的值作为输入参数对variable进行更新
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    summary_op = tf.summary.merge_all()
    # 滑动平均模型的作用是提高测试值上的健壮性, 模型参数进行平均得到的模型往往比单个模型的结果要好很多
    # decay = min(decay, (1 + num_updates) / (10 + num_updates))
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        # tf.no_op()表示执行完 variables_averages_op, apply_gradient_op, batch_norm_updates_op操作之后什么都不做
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.max_to_keep)
    summary_writer = tf.summary.FileWriter(FLAGS.logs_path + styletime, tf.get_default_graph())

    exclude = ['global_step']
    init_except_densenet_parameter = slim.get_variables_to_restore(exclude=exclude)  # 加载预训练模型
    init = tf.global_variables_initializer()

    if FLAGS.pretrained_model_path is not None:
        variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path,
                                                             init_except_densenet_parameter,
                                                             ignore_missing_vars=True)
    else:
        variable_restore_op = None
    # 定义训练数据获取op
    img_data, shape, lable_seq, lable_seq_len, image_w = \
        generator_tool.GeneratorTfrecordTool.read_slice_gfile_synthetic_chinese_string_dataset_tfrecord_data(
            FLAGS.tfrecord_path)

    # targets_lable = generator_tool.dense2sparse(lable_seq)

    # indices_t, values_t, shape_t = util.sparse_tuple_from(lable_seq)
    # targets_lable = util.get_sparsetensor(indices_t, values_t, shape_t)  # 获取标签稀疏矩阵

    # max_steps = generator_tool_bak.GeneratorTfrecordTool.total_sample(FLAGS.tfrecord_path)  # 有多少个数据
    max_steps = generator_tool.GeneratorTfrecordTool.total_sample_by_path(FLAGS.tfrecord_path)  # 有多少个数据
    max_steps = max_steps // 5  # 采样5个
    print("max_steps:", max_steps)
    coord = tf.train.Coordinator()
    # with tf.Session(config=config) as sess:
    with tf.Session() as sess:
        queue_runner = tf.train.start_queue_runners(sess, coord=coord)
        if FLAGS.pretrained_model_path is not None:
            variable_restore_op(sess)

        if FLAGS.restore:  # 是否继续训练
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            # ocr_100_1.ckpt
            restore_step = int(ckpt.split('.')[0].split('_')[-1])
            if restore_step + 1 >= max_steps:
                restore_step = 0
            print("continue training from previous checkpoint {}".format(restore_step))
            saver.restore(sess, ckpt)
        else:
            sess.run(init)
            restore_step = 0
            print("init checkpoint {}".format(restore_step))

        '''
        # 切片
        input_queue = tf.train.slice_input_producer(
            [img_data_, shape_, lable_seq_, lable_seq_len_, image_w_], shuffle=False)
        # 按顺序读取队列中的数据,batch_size要和占位符中的batch_size一样
        img_data_t, shape_t, lable_seq_t, lable_seq_len_t, image_w_t = tf.train.batch(
            input_queue, batch_size=BATCH_SIZE, num_threads=1, capacity=64)
        print("type:", type(img_data_t), type(lable_seq_t), type(image_w_t))
        '''
        '''
        img_data_t = img_data_
        targets_lable_t = targets_lable
        image_w_t = image_w
        '''

        for epoch in range(FLAGS.training_epochs):
            print("epoch:", epoch)
            for step in range(restore_step, max_steps):

                # 得到训练数据,必须放在循环里面
                img_data_, shape_, lable_seq_, lable_seq_len_, image_w_ = sess.run(
                    [img_data, shape, lable_seq, lable_seq_len, image_w])

                # print(type(img_data_), type(shape_), type(lable_seq_), type(lable_seq_len_), type(image_w_))

                # print("step:", step)
                # print(lable_seq_.shape)
                lable_text = [lable_map[_] for _ in lable_seq_[0]]
                # print("真实数据:", "".join(lable_text), lable_seq_[0])

                # feed 数据
                tloss, decoded_t, _, summary_str = sess.run([total_loss, decoded_test, train_op, summary_op],
                                                            feed_dict={input_image: img_data_,
                                                            targets: lable_seq_,
                                                            seq_len: image_w_})
                # print(tloss)
                summary_writer.add_summary(summary_str, global_step=step)

                decoded_dense = tf.sparse_to_dense(decoded_t[0].indices, decoded_t[0].dense_shape, decoded_t[0].values,
                                                   default_value=-1)
                # print("decoded_dense:", type(decoded_dense.eval()), decoded_dense)
                label_predicts_idx = [_ for _ in decoded_dense.eval()[0]]
                label_predicts = [lable_map[_] for _ in label_predicts_idx]
                # print("结果显示:", ''.join(label_predicts))

                if step != 0 and step % FLAGS.decay_steps == 0:
                    if learning_rate.eval() * FLAGS.decay_rate >= 1e-10:
                        sess.run(tf.assign(learning_rate, learning_rate.eval() * FLAGS.decay_rate))  # 学习率更新
                    # print("learning_rate:", learning_rate.eval() * FLAGS.decay_rate)
                    pass

                if step % 100 == 0:
                    print('Epoch {:d} Step {:06d}, model loss {:.4f}, LR: {:.10f}'.
                          format(epoch, step, tloss, learning_rate.eval()))
                    print("真实数据:", "".join(lable_text))
                    print("结果显示:", ''.join(label_predicts))

                if step % 1000 == 0:  # 1000张图片或每迭代10轮保存一次模型
                    filename = ('ocr_' + '{:d}'.format(epoch) + '_' + '{:d}'.format(step//1000) + '.ckpt')
                    filename = os.path.join(FLAGS.checkpoint_path, filename)
                    saver.save(sess, filename)
                    print('Write model to: {:s}'.format(filename))

        coord.request_stop()
        coord.join(queue_runner)


if __name__ == '__main__':
    print("nohup /home/gaofangjie/tvenv/bin/python3 train.py > /home/gaofangjie/train_data.txt 2>&1&")
    tf.app.run()
