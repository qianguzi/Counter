#import glob
import os
import time

import tensorflow as tf

from config import *
from mobilenetv2 import mobilenetv2
from utils import get_batch, load


def main():
    #height = args.height
    #width = args.width
    sess = tf.Session()

    # read queue
    glob_pattern = os.path.join(args.dataset_dir, 'train.tfrecords')
    #tfrecords_list = glob.glob(glob_pattern)

    img_batch, label_batch, _ = get_batch(
        glob_pattern, args.batch_size, args.epoch)
    # print(img_batch)

    #inputs = tf.placeholder(tf.float32, [None, height, width, 3], name='input')

    logits, pred, _ = mobilenetv2(
        img_batch, wid=args.wid, num_classes=args.num_classes, is_train=args.is_train)

    # loss
    loss_ = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_batch, logits=logits))
    # L2 regularization
    l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    total_loss = loss_ + l2_loss

    # evaluate model, for classification
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.cast(label_batch, tf.int64))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # learning rate decay
    base_lr = tf.constant(args.learning_rate)
    lr_decay_step = args.num_samples // args.batch_size * 2  # every epoch
    global_step = tf.placeholder(dtype=tf.float32, shape=())
    lr = tf.train.exponential_decay(base_lr, global_step=global_step, decay_steps=lr_decay_step,
                                    decay_rate=args.lr_decay)
    # optimizer
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        #tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=0.9, momentum=0.9)
        train_op = tf.train.AdamOptimizer(
            learning_rate=lr, beta1=args.beta1).minimize(total_loss)

    # summary
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('accuracy', acc)
    tf.summary.scalar('learning_rate', lr)
    summary_op = tf.summary.merge_all()

    # summary writer
    writer = tf.summary.FileWriter(args.logs_dir, sess.graph)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    step = 0

    # saver for save/restore model
    saver = tf.train.Saver(max_to_keep=15)
    # load pretrained model
    if not args.renew:
        print('[*] Try to load trained model...')
        load(sess, saver, args.checkpoint_dir)

    max_steps = int(args.num_samples / args.batch_size)
    _epoch = 1
    print(time.strftime("%X"), 'START TRAINING...')
    try:
        while not coord.should_stop():
            print('epoch:%d/%d' % (_epoch, args.epoch))
            for _step in range(step+1, step+max_steps+1):
                #start_time = time.time()
                gl_step = (_epoch - 1) * max_steps + _step
                feed_dict = {global_step: gl_step}
                # train
                _, _lr, _summ, _loss, _acc = sess.run([train_op, lr, summary_op, total_loss, acc],
                                                      feed_dict=feed_dict)

                # print logs 、write summary 、save model
                if _step % 20 == 0:
                    writer.add_summary(_summ, _step)
                    print(time.strftime("%X"), 'global_step:{0}, lr:{1:.8f}, acc:{2:.6f}, loss:{3:.6f}'
                          .format(gl_step, _lr, _acc, _loss))
            save_path = saver.save(sess, os.path.join(
                args.checkpoint_dir, args.model_name), global_step=_epoch)
            print('Current model saved in ' + save_path)
            _epoch += 1
    except tf.errors.OutOfRangeError:
        tf.train.write_graph(
            sess.graph_def, args.checkpoint_dir, args.model_name + '.pb')
    finally:
        coord.request_stop()
        coord.join(threads)
        sess.close()
    print('FINISHED TRAINING.')


if __name__ == '__main__':
    main()
