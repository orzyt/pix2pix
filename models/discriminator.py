import tensorflow as tf
from ops import lrelu


class Discriminator(object):
    def __init__(self, inputs, is_training=True, ndf=64, reuse=False):
        self.ndf = ndf
        self.inputs = inputs
        self.is_training = is_training
        self.logits = self.build_discriminator(inputs, reuse=reuse)

    def build_layers(self, inputs, filters, stride, use_bn=True, activation=None, name=None):
        padded_input = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        with tf.variable_scope(name):
            x = tf.layers.conv2d(padded_input, filters, kernel_size=4, strides=(stride, stride), padding='VALID',
                                 kernel_initializer=tf.truncated_normal_initializer(0, 0.02))
            if use_bn:
                x = tf.layers.batch_normalization(x, training=self.is_training)
            if activation is not None:
                x = activation(x)
            return x

    def build_discriminator(self, inputs, reuse=False, name='discriminator'):
        # C64-C128-C256-C512
        with tf.variable_scope(name, reuse=reuse):
            # shape: 128 * 128 * 64
            d1 = self.build_layers(inputs, self.ndf, 2, use_bn=False, activation=lrelu, name='d1')
            # shape: 64 * 64 * 128
            d2 = self.build_layers(d1, self.ndf * 2, 2, activation=lrelu, name='d2')
            # shape: 32 * 32 * 256
            d3 = self.build_layers(d2, self.ndf * 4, 2, activation=lrelu, name='d3')
            # shape: 31 * 31 * 512
            d4 = self.build_layers(d3, self.ndf * 8, 1, activation=lrelu, name='d4')
            # shape: 30 * 30 * 1
            d5 = self.build_layers(d4, 1, 1, use_bn=False, activation=tf.nn.sigmoid, name='d5')
        return d5
