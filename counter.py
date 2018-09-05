import os
import codecs, json
from time import strftime, time

import cv2
import numpy as np

from CapDetection import CapDetection
from mobilenetv2 import mobilenetv2
from ops import *

if __name__ == '__main__':
    # bulid model
    inputs = tf.placeholder(tf.float32, [None, 64, 64, 3], name='input')
    _, pred, _ = mobilenetv2(inputs, wid=3, num_classes=2, is_train=False)
    y = tf.argmax(pred, 1)

    ckpt = '../checkpoints/64x64-128-0.95-0.001-wid3/mobilenetv2-17'
    sess = tf.Session()
    # saver for restore model
    saver = tf.train.Saver()
    saver.restore(sess, ckpt)

    img_dir = '/media/jun/data/capdataset/detection/test/'
    result_dir = '/media/jun/data/capdataset/detection/result_test/'
    cirfile = '/media/jun/data/capdataset/detection/result_test.json'
    cap_types = [(6, 10), (5, 8), (6, 10), (6, 10), (5, 10)]
    boxes = CapDetection(img_dir, result_dir)
    result_boxes = {}
    for img_name in boxes.imgs:
        t = int(img_name[0])
        print(strftime("%X"), img_name + '检测开始...')
        start_time = time()
        ###################检测############################
        pre_img = boxes.preprocessing(boxes.imgs[img_name])
        cir_dense = boxes.hough(pre_img, 50, 30)
        cir_sparse = boxes.hough(pre_img, 60, 35)
        grid, crop_size = boxes.grid(cir_sparse, cap_types[t])
        color_img = cv2.cvtColor(boxes.imgs[img_name], cv2.COLOR_GRAY2RGB)
        can_dense, mark_dense = boxes.get_canimgs(color_img, cir_dense)
        can_sparse, mark_sparse = boxes.get_canimgs(color_img, grid, crop_size)

        dec_time = time()
        print('检测所用时间: %s' % (dec_time-start_time))

        ##################dense结果####################
        x_dense = can_dense / 127.5 - 1
        dense_pred = sess.run(y, feed_dict={inputs: x_dense})
        dense_pred = np.array(dense_pred)
        result_dense = mark_dense[np.where(dense_pred == 1)]

        ##################sparse结果##################################
        x_sparse = can_sparse / 127.5 - 1
        sparse_pred = sess.run(y, feed_dict={inputs: x_sparse})
        sparse_pred = np.array(sparse_pred)
        result_sparse = mark_sparse[np.where(sparse_pred == 1)]

        rec_time = time()
        print('识别所用时间: %s' % (rec_time-dec_time))
        ####################结果##################################
        #can_imgs = np.vstack((can_sparse, can_dense))
        #can_marks = np.vstack((mark_sparse, mark_dense))
        #x = can_imgs / 127.5 - 1
        #_pred = sess.run(y, feed_dict={inputs: x})
        #_pred = np.array(_pred)
        #result = can_marks[np.where(_pred==1)]

        result_cir = boxes.hough(pre_img, 60, 45)
        result_cir = np.array(result_cir[:, 0:2])
        result = np.concatenate(
            (result_sparse, result_dense, result_cir), axis=0)
        ###################final#####################################
        same = []
        for n, i in enumerate(result[:-1]):
            for j in result[n+1:]:
                if abs(i[0]-j[0]) <= crop_size[0] and abs(i[1]-j[1]) <= crop_size[1]:
                    same.append(n)
                    break
        final = np.delete(result, same, 0)
        #########################################################
        end_time = time()
        print('筛选所用时间: %s' % (end_time-rec_time))
        print('所用总时间: %s' % (end_time-start_time))
        print(strftime("%X"), img_name + '检测成功...')

        ##################写入结果##################################
        #result_img = boxes.draw_cir(
        #    color_img, result_dense, rad=35, color=(0, 0, 255))
        #result_img = boxes.draw_cir(
        #    result_img, result_sparse, rad=25, color=(255, 0, 0))
        #result_img = boxes.draw_cir(
        #    result_img, result_cir, rad=15, color=(0, 255, 0))
        #cv2.imwrite(result_dir + img_name[:-4] + 'result.jpg', result_img)
        #
        ################写入最终结果##############################
        final_img, gt_boxes = boxes.draw_rec(boxes.imgs[img_name], final, t, crop_size)
        #cv2.imwrite(result_dir + img_name[:-4] + 'final.jpg', final_img)
        result_boxes[img_name[:-4]] = gt_boxes
    json.dump(result_boxes, codecs.open(cirfile, 'w', encoding='utf-8'), separators=(',', ':'), indent=4)
