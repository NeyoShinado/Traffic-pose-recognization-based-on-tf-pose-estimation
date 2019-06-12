import tensorflow as tf
import xlrd
import random
import numpy as np
import time


def get_Batch(file, num_s, epochs=600, batch_size=300, shuffle=True):
    # file: file of xlsx data
    # num_s: number of sample for training
    # epochs: number of dataset retrain rounds
    # batch_size: number of samples per batch
    # shuffle: either shuffle the samples or not
    wb = xlrd.open_workbook(file)
    sheet = wb.sheets()[0]
    labels = [[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]]
    poses = ["go_straight", "park_right", "stop", "turn_right"]

    shu_id = random.sample(range(0, sheet.nrows), sheet.nrows)
    data = np.empty((num_s, 36), dtype=np.float32)
    label = np.empty((num_s, 4), dtype=np.float32)
    for i in range(num_s):
        if shuffle:
            data[i] = sheet.row_values(shu_id[i])[:36]
            idx = poses.index(sheet.row_values(shu_id[i])[-1])
            label[i] = labels[idx]
        else:
            data[i] = sheet.row_values(i)[:36]
            idx = poses.index(sheet.row_values(i)[-1])
            label[i] = labels[idx]

    print(data.shape, label.shape)
    input_queue = tf.train.slice_input_producer([data, label], num_epochs=epochs, shuffle=True, capacity=64)
    x_batch, y_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=64, allow_smaller_final_batch=False)
    return x_batch, y_batch, data, label


def Tp_training(epochs, batch_size, tr_kp):
    t_start = time.time()
    data_file = "../../traffic_pose/keypoint_data/training_data.xlsx"
    result = get_Batch(data_file, 3300, epochs=epochs, batch_size=batch_size,
                                                          shuffle=True)
    x_batch, y_batch = result[0], result[1]
    result = get_Batch(data_file, 73, epochs=epochs, batch_size=batch_size,
                                                          shuffle=True)
    test_data, test_label = result[2], result[3]
    t_batch = time.time() - t_start
    print("___batching finish in %.4f s___" % t_batch)

    # define graph
    in_units = 36
    h1_units = 32
    w1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1), name='w1')
    b1 = tf.Variable(tf.zeros([h1_units]), name='b1')
    w2 = tf.Variable(tf.zeros([h1_units, 4]), name='w2')
    b2 = tf.Variable(tf.zeros([4]), name='b2')

    x = tf.placeholder(tf.float32, [None, in_units], name='x')
    keep_prob = tf.placeholder(tf.float32)

    hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
    y = tf.nn.softmax(tf.matmul(hidden1_drop, w2) + b2)

    y_ = tf.placeholder(tf.float32, [None, 4], name='gnd')
    loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    optimizer = tf.train.AdagradOptimizer(0.3).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # training
    t_train = time.time()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess, coord)
        epoch = 0
        try:
            while not coord.should_stop():
                data, label = sess.run([x_batch, y_batch])
                sess.run(optimizer, feed_dict={x: data, y_: label, keep_prob: tr_kp})
                train_accuracy = accuracy.eval({x: data, y_: label, keep_prob: tr_kp})
                test_accuracy = accuracy.eval({x: test_data, y_: test_label, keep_prob: 1})
                print("___Epoch %d Training accuracy %g, Testing accuracy %g___" % (epoch, train_accuracy, test_accuracy))
                epoch += 1
        except tf.errors.OutOfRangeError:
            print("___Training end___")

        finally:
            coord.request_stop()

        t_end = time.time() - t_train
        print("___Programme end in %.4f s___" % t_end)
        coord.join(threads)

        # save model
        saver.save(sess, "../models/Pose_recg/pr")


if __name__ == "__main__":
    Tp_training(1500, 300, 0.75)