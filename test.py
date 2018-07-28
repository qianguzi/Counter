#import glob
import os
import time

import tensorflow as tf

from config import *
from mobilenetv2 import mobilenetv2
from utils import get_batch, load


def main():
    errfile = '../_error/other_errors.txt'
    num_samples = 14292

    sess = tf.Session()

    # read queue
    glob_pattern = os.path.join(args.dataset_dir, 'other.tfrecords')
    #tfrecords_list = glob.glob(glob_pattern)

    img_batch, label_batch, name_batch = get_batch(
        glob_pattern, args.batch_size, epochs=1, shuffle=False)
    #inputs = tf.placeholder(tf.float32, [None, height, width, 3], name='input')

    _, pred, _ = mobilenetv2(img_batch, wid=args.wid,
                             num_classes=args.num_classes, is_train=False)

    # evaluate model, for classification
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.cast(label_batch, tf.int64))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # saver for restore model
    saver = tf.train.Saver()
    print('[*] Try to load trained model...')
    ckpt_name = load(sess, saver, args.checkpoint_dir)

    step = 0
    accs = 0
    me_acc = 0
    errors_name = []
    max_steps = int(num_samples / args.batch_size)
    print('START TESTING...')
    try:
        while not coord.should_stop():
            for _step in range(step+1, step+max_steps+1):
                # test
                _name, _corr, _acc = sess.run([name_batch, correct_pred, acc])
                if (~_corr).any():
                    errors_name.extend(list(_name[~_corr]))
                accs += _acc
                me_acc = accs/_step
                if _step % 20 == 0:
                    print(time.strftime("%X"), 'global_step:{0}, current_acc:{1:.6f}'.format
                          (_step, me_acc))
    except tf.errors.OutOfRangeError:
        accuracy = 1 - len(errors_name)/num_samples
        print(time.strftime("%X"),
              'RESULT >>> current_acc:{0:.6f}'.format(accuracy))
        # print(errors_name)
        errorsfile = open(errfile, 'a')
        errorsfile.writelines('\n' + ckpt_name + '--' + str(accuracy))
        for err in errors_name:
            errorsfile.writelines('\n' + err.decode('utf-8'))
        errorsfile.close()
    finally:
        coord.request_stop()
        coord.join(threads)
        sess.close()
    print('FINISHED TESTING.')


if __name__ == '__main__':
    main()
