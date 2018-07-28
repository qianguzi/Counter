from ops import *


def mobilenetv2(inputs, num_classes, wid=2, is_train=True, reuse=False):
    exp = 6  # expansion ratio
    with tf.variable_scope('mobilenetv2'):
        net = conv2d_block(inputs, 8*wid, 3, 2, is_train,
                           name='conv1_1')  # size/2

        net = bottleneck(net, 1, 4*wid, 1, is_train, name='res2_1')

        net = bottleneck(net, exp, 6*wid, 2, is_train, name='res3_1')  # size/4
        net = bottleneck(net, exp, 6*wid, 1, is_train, name='res3_2')

        net = bottleneck(net, exp, 8*wid, 2, is_train, name='res4_1')  # size/8
        net = bottleneck(net, exp, 8*wid, 1, is_train, name='res4_2')
        net = bottleneck(net, exp, 8*wid, 1, is_train, name='res4_3')

        net = bottleneck(net, exp, 16*wid, 2, is_train, name='res5_1')  # size/16
        net = bottleneck(net, exp, 16*wid, 1, is_train, name='res5_2')
        net = bottleneck(net, exp, 16*wid, 1, is_train, name='res5_3')
        net = bottleneck(net, exp, 16*wid, 1, is_train, name='res5_4')

        net = bottleneck(net, exp, 24*wid, 1, is_train, name='res6_1')
        net = bottleneck(net, exp, 24*wid, 1, is_train, name='res6_2')
        net = bottleneck(net, exp, 24*wid, 1, is_train, name='res6_3')

        net = bottleneck(net, exp, 40*wid, 2, is_train, name='res7_1')  # size/32
        net = bottleneck(net, exp, 40*wid, 1, is_train, name='res7_2')
        net = bottleneck(net, exp, 40*wid, 1, is_train, name='res7_3')

        net = bottleneck(net, exp, 80*wid, 1, is_train,
                         name='res8_1', shortcut=False)

        net = point_wise(net, 320*wid, is_train, name='conv9_1')
        net = global_avg(net)
        logits = flatten(conv_1x1(net, num_classes, name='logits'))

        pred = tf.nn.softmax(logits, name='prob')
        return logits, pred, net
