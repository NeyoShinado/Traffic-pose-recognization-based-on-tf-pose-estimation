#!/usr/bin/env python
#-------------------- Traffic Pose recognition ------------------#
#-------------------------- Neyo,2019 ---------------------------#
#---------- JPG files oriented with name "pose%index" -----------#
#----------- Files size should be lower than 500*500 ------------#

import argparse
import logging
import time
import glob
import ast
import xlsxwriter
import numpy as np
import tensorflow as tf
import tf_pose.common as common
import cv2
import xlrd
from tf_pose.estimator import TfPoseEstimator, Human
from tf_pose.networks import get_graph_path, model_wh
from Traffic_pose.data_process import data_reshape
from Traffic_pose.fig_resize import resize_data
from Traffic_pose.Pose_recognizer import Tp_training, get_Batch

#from lifting.prob_model import Prob3dPose
#from lifting.draw import plot_pose


logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# humans -- list of human object
# all_humans -- dict for four dicts of humans per image
# human.body_parts -- dict of part_CoCo & coordinates


def point_extract(dict, out_dir):
    workbook = xlsxwriter.Workbook(out_dir)
    worksheet = workbook.add_worksheet()

    row = 0
    for file, humans in dict.items():
    # write humans key_points
        for human_idx in range(len(humans)):
            part = str(humans[human_idx].body_parts).split('BodyPart')        # write a human's key point
            worksheet.write_string(row, 0, file[1:])         # write head_of image
            try:
                for i in part[1:]:
                    body, loc = i.split('-')[0], i.split('-')[1]
                    worksheet.write_string(row, int(body[1:])+1, loc[1:11])

            except Exception as e:
                print(e)

            row += 1
            print("___write #%d human from file: %s done___" %(human_idx, file))
    workbook.close()


def pose_estimate(data_folder, args):
    # files_grabbed = glob.glob(os.path.join(args.folder, '*.png'))
    files_grabbed = glob.glob(data_folder + '*.jpg')
    all_humans = {"go_straight":{}, "park_right":{}, "stop":{}, "turn_right":{}}

    # getting correct output idx for link
    name_idx = files_grabbed[0].find(data_folder[-4:]) + 4

    for i, file in enumerate(files_grabbed):
        # estimate human poses from a single image
        print("___Estimate #%dth fig's human pose___" %i)
        class_name = file.split("\\")[3].split("%")[0]
        image = common.read_imgfile(file, None, None)
        t = time.time()

        # humans = e.inference(image, scales=scales)
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        elapsed = time.time() - t

        logger.info('inference image #%d: %s in %.4f seconds.' % (i, file, elapsed))

        # draw skeletal
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        cv2.imwrite(r".\images\result_%s\%s" % (class_name, str(file[name_idx:])), image)
        cv2.waitKey(5)

        all_humans[class_name][file.replace(args.folder, '')] = humans

    # write human keypoint
    for pose in all_humans.keys():
        data_dir = data_folder[:15] + '\keypoint_data\\' + pose + '.xlsx'
        point_extract(all_humans[pose], data_dir)

    print("___write all humans' keypoint done___")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='tf-pose-estimation run by folder')
    parser.add_argument('--folder', type=str, default='.\images')
    parser.add_argument('--resolution', type=str, default='432*368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--scales', type=str, default='[None]', help='for multiple scales, eg. [1.0, (1.1, 0.05)]')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    args = parser.parse_args()
    scales = ast.literal_eval(args.scales)

    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    ## Dataset prepose
    data_folder = '..\\traffic_pose\\src\\'
    poses = ['go_straight', 'park_right', 'stop', 'turn_right']
    resize_data(data_folder)

    ## Pose estimation
    #data_folder = '..\\traffic_pose\\%s_new\\' %(pose)
    print("___Extracting figures form source foder: %s___" % data_folder)
    pose_estimate(data_folder, args)


    # extract keypoints' x, y coordinary
    src = "../traffic_pose/keypoint_data/"
    data_reshape(src, poses)

    ## Training MLP model for pose recognize
    #Tp_training(1000, 300, 8.0)

    ## Building test data
    data_file = "../traffic_pose/keypoint_data/training_data.xlsx"
    wb = xlrd.open_workbook(data_file)
    sheet = wb.sheets()[0]
    num_s = sheet.nrows
    result = get_Batch(data_file, num_s, shuffle=False)
    testdata, testlabel = result[2], result[3]

    ## Pose Recognization
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("./models/Pose_recg/pr.meta")
        saver.restore(sess, tf.train.latest_checkpoint('./models/Pose_recg/'))

        # model for test
        graph = tf.get_default_graph()
        x = tf.placeholder(tf.float32, [None, 36])
        gnd = tf.placeholder(tf.float32, [None, 4])

        w1 = graph.get_tensor_by_name("w1:0")
        w2 = graph.get_tensor_by_name("w2:0")
        b1 = graph.get_tensor_by_name("b1:0")
        b2 = graph.get_tensor_by_name("b2:0")
        hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1)
        y = tf.nn.softmax(tf.matmul(hidden1, w2) + b2)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(gnd, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #pred = sess.run(tf.argmax(y, 1), feed_dict=({x: testdata}))
        test_accuracy = accuracy.eval({x: testdata, gnd: testlabel})
        print("___Testing accuracy %g___" % test_accuracy)

        score = sess.run(y, feed_dict=({x: testdata}))              # predict score of human
        print("score of all humans:\n", score)

        # save pred result
        pre_class = []
        for i in range(len(score)):
            pre_class.append(poses[score[i].tolist().index(max(score[i]))])                        # predict class of human
        print("pre_class of all humans:\n", pre_class)
        workbook = xlsxwriter.Workbook("../traffic_pose/keypoint_data/testresult.xlsx")
        worksheet = workbook.add_worksheet()
        for i, item in enumerate(pre_class):
            worksheet.write(i, 0, item)
        workbook.close()

    ## drawing result
    filelist = sheet.col_values(36)
    file_name = set(filelist)
    for file in file_name:
        startid = filelist.index(file)
        endid = max([i for i, x in enumerate(filelist) if x == file])
        human_row, class_id = np.where(score[startid:endid+1] == np.max(score[startid:endid+1]))  # opt human detector
        print("___file #%s: policeman #%d, class #%s___" % (file, human_row, poses[class_id[0]]))

        image = common.read_imgfile(data_folder + file, None, None)
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        box = humans[human_row[0]].get_upper_body_box(np.shape(image)[1], np.shape(image)[0])
        try:
            x = box['x']
            y = box['y']
            w = box['w']
            h = box['h']
            cv2.rectangle(image, (x-round(w/2), y-round(h/2)), (x+round(w/2), y+round(h/2)), [4,250,7], 4)
            Font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, poses[class_id[0]], (x-round(w/2), y-round(h/2)), Font, 0.8, [4, 250, 7], 2)
        except e:
            print(e)
        cv2.imwrite(r"..\traffic_pose\result\%s" %file, image)
        print("___tag figure #:%s done___" % file)
        cv2.waitKey(5)
