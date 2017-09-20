import tensorflow as tf


def linear(inpt,
           output_size,
           stddev=0.02,
           bias_start=0.0,
           activation_fn=None,
           name='linear'):
    shape = inpt.get_shape().as_list()
    with tf.variable_scope(name):
        w = tf.get_variable('w', [shape[1], output_size], tf.float32,
                            tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_size],
                            initializer=tf.constant_initializer(bias_start))

        out = tf.nn.bias_add(tf.matmul(inpt, w), b)

        if activation_fn is not None:
            return activation_fn(out)
        else:
            return out


def conv2d(inpt,
           output_dim,
           kernel_size,
           stride,
           initializer=tf.contrib.layers.xavier_initializer(),
           activation_fn=tf.nn.relu,
           data_format='NHWC',
           padding='VALID',
           name='conv2d'):
    with tf.variable_scope(name):
        if data_format == 'NCHW':
            stride = [1, 1, stride[0], stride[1]]
            kernel_shape = [kernel_size[0], kernel_size[1], inpt.get_shape()[1], output_dim]
        elif data_format == 'NHWC':
            stride = [1, stride[0], stride[1], 1]
            kernel_shape = [kernel_size[0], kernel_size[1], inpt.get_shape()[-1], output_dim]

        w = tf.get_variable('w', kernel_shape, tf.float32, initializer=initializer)
        conv = tf.nn.conv2d(inpt, w, stride, padding, data_format=data_format)

        b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, b, data_format)

    if activation_fn is not None:
        out = activation_fn(out)

    return out


def mlp(inpt, num_outputs, hiddens=[20], activation_fn=tf.nn.relu):
    out = inpt
    for i, hidden in enumerate(hiddens):
        out = linear(out, num_outputs, activation_fn=activation_fn, name='l_'+str(i))
    out = linear(out, num_outputs, name='out')
    return out


def deepmind_CNN(inpt, num_outputs):
    initializer = tf.truncated_normal_initializer(0, 0.1)
    activation_fn = tf.nn.relu
    # inpt = tf.transpose(inpt, [0, 2, 3, 1])
    l1 = conv2d(inpt, 32, [8, 8], [4, 4],
                initializer, activation_fn, 'NHWC', name='l1')
    l2 = conv2d(l1, 64, [4, 4], [2, 2],
                initializer, activation_fn, 'NHWC', name='l2')
    l3 = conv2d(l2, 64, [3, 3], [1, 1],
                initializer, activation_fn, 'NHWC', name='l3')

    shape = l3.get_shape().as_list()
    l3_flat = tf.reshape(l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

    out = mlp(l3_flat, num_outputs, [128])

    return out
