import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image


# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 生成字符串类型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_tfrecord(filename, mapfile, dataset_path):
    ''' Covert Image dataset to tfrecord. '''

    class_map = {}
    classes = ['0', '1']
    writer = tf.python_io.TFRecordWriter(filename)

    for index, class_name in enumerate(classes):
        class_path = dataset_path + class_name + '/'
        if index == 2:
            index = 1
        class_map[index] = class_name
        for img_name in os.listdir(class_path):
            img = Image.open(class_path + img_name)
            img = img.convert("RGB")
            img = img.resize((64, 64))
            #img = img.resize((64, 64))
            # print(np.array(img).shape)
            img_raw = img.tobytes()
            img_name = img_name.encode()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(index),
                'image_raw': _bytes_feature(img_raw),
                'image_name': _bytes_feature(img_name)
            }))
            writer.write(example.SerializeToString())
    writer.close()
    txtfile = open(mapfile, 'w+')
    for key in class_map.keys():
        txtfile.writelines(str(key)+":"+class_map[key]+"\n")
    txtfile.close()


def read_tfrecord(filename, epochs, shuffle=True):
    # 创建一个reader来读取TFRecord文件中的样例
    reader = tf.TFRecordReader()
    # 创建一个队列来维护输入文件列表
    filename_queue = tf.train.string_input_producer(
        [filename], shuffle=shuffle, num_epochs=epochs)
    # 从文件中读出一个样例，也可以使用read_up_to一次读取多个样例
    _, serialized_example = reader.read(filename_queue)

    # 解析读入的一个样例，如果需要解析多个，可以用parse_example
    features = tf.parse_single_example(
        serialized_example,
        features={'label': tf.FixedLenFeature([], tf.int64),
                  'image_raw': tf.FixedLenFeature([], tf.string),
                  'image_name': tf.FixedLenFeature([], tf.string)})
    # 将字符串解析成图像对应的像素数组
    img = tf.decode_raw(features['image_raw'], tf.uint8)
    img = tf.reshape(img, [64, 64, 3])
    #img = tf.reshape(img, [64, 64, 3])
    #img = tf.cast(img, tf.float32)*1/255 - 0.5
    img = tf.cast(img, tf.float32)*1/127.5 - 1
    label = tf.cast(features['label'], tf.int32)
    img_name = tf.cast(features['image_name'], tf.string)

    return img, label, img_name


def get_batch(filename, batch_size, epochs, num_threads=4, shuffle=True, min_after_dequeue=None):
    '''Get batch.'''
    # 使用batch，img的shape必须是静态常量
    image, label, image_name = read_tfrecord(filename, epochs, shuffle)

    if min_after_dequeue is None:
        min_after_dequeue = batch_size*10
    #capacity = min_after_dequeue + 3 * batch_size
    capacity = 60000

    if shuffle:
        img_batch, label_batch, name_batch = tf.train.shuffle_batch([image, label, image_name], batch_size=batch_size,
                                                                    capacity=capacity, num_threads=num_threads,
                                                                    min_after_dequeue=min_after_dequeue)
    else:
        img_batch, label_batch, name_batch = tf.train.batch([image, label, image_name], batch_size,
                                                            capacity=capacity, num_threads=num_threads,
                                                            allow_smaller_final_batch=True)

    return img_batch, label_batch, name_batch


def load(sess, saver, checkpoint_dir):
    #import re
    print("[*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        # print(ckpt_name)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        #counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print("[*] Success to read {}".format(ckpt_name))
        return ckpt_name
    else:
        raise Exception("[*] Failed to find a checkpoint")
       # return False


if __name__ == '__main__':
    mapfile = '../tfrecords/other_classmap.txt'
    filename = '../tfrecords/other.tfrecords'
    dataset_path = '/media/jun/data/capdataset/cap/others/'
    create_tfrecord(filename, mapfile, dataset_path)
