import tensorflow as tf
from ops import lrelu


class Generator(object):
    def __init__(self, inputs, is_training=True, ngf=64, out_channels=3, reuse=False):
        self.ngf = ngf
        self.inputs = inputs
        self.channels = out_channels
        self.is_training = is_training
        self.outputs = self.build_unet(inputs, reuse=reuse)

    def build_encoder_layers(self, inputs, filters, use_bn=True, name=None):
        with tf.variable_scope(name):
            x = tf.layers.conv2d(inputs, filters, kernel_size=4, strides=(2, 2), padding='SAME',
                                 kernel_initializer=tf.truncated_normal_initializer(0, 0.02))
            if use_bn:
                x = tf.layers.batch_normalization(x, training=self.is_training)
            x = lrelu(x)
            return x

    def build_decoder_layers(self, inputs, filters, prob=0.0, use_bn=True, activation=None, name=None):
        with tf.variable_scope(name):
            x = tf.layers.conv2d_transpose(inputs, filters, kernel_size=4, strides=(2, 2), padding='SAME')
            if use_bn:
                x = tf.layers.batch_normalization(x, training=self.is_training)
            if prob > 0.0:
                x = tf.layers.dropout(x, prob)
            if activation is not None:
                x = activation(x)
            return x

    def build_encoder(self, inputs, name='encoder'):
        with tf.variable_scope(name):
            # C64-C128-C256-C512-C512-C512-C512-C512
            config = [
                self.ngf * 1,  # encoder_1: [batch, 256, 256, in_channels]  =>  [batch, 128, 128, ngf]
                self.ngf * 2,  # encoder_2: [batch, 128, 128, ngf]          =>  [batch, 64, 64, ngf * 2]
                self.ngf * 4,  # encoder_3: [batch, 64, 64, ngf * 2]        =>  [batch, 32, 32, ngf * 4]
                self.ngf * 8,  # encoder_4: [batch, 32, 32, ngf * 4]        =>  [batch, 16, 16, ngf * 8]
                self.ngf * 8,  # encoder_5: [batch, 16, 16, ngf * 8]        =>  [batch, 8, 8, ngf * 8]
                self.ngf * 8,  # encoder_6: [batch, 8, 8, ngf * 8]          =>  [batch, 4, 4, ngf * 8]
                self.ngf * 8,  # encoder_7: [batch, 4, 4, ngf * 8]          =>  [batch, 2, 2, ngf * 8]
                self.ngf * 8,  # encoder_8: [batch, 2, 2, ngf * 8]          =>  [batch, 1, 1, ngf * 8]
            ]
            encoder = []
            for filter in config:
                n = len(encoder)
                input = encoder[-1] if n > 0 else inputs
                use_bn = True if (n > 0 and n != len(config) - 1) else False
                encoder.append(self.build_encoder_layers(input, filter, use_bn, 'encoder_%d' % (n + 1)))
        return encoder

    def build_decoder(self, encoder, name='decoder'):
        with tf.variable_scope(name):
            # CD512-CD512-CD512-C512-C256-C128-C64
            # U-Net: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
            config = [
                (self.ngf * 8, 0.5),  # decoder_8: [batch, 1, 1, ngf * 8]       =>  [batch, 2, 2, ngf * 8]
                (self.ngf * 8, 0.5),  # decoder_7: [batch, 2, 2, ngf * 8]       =>  [batch, 4, 4, ngf * 8]
                (self.ngf * 8, 0.5),  # decoder_6: [batch, 4, 4, ngf * 8]       =>  [batch, 8, 8, ngf * 8]
                (self.ngf * 8, 0.0),  # decoder_5: [batch, 8, 8, ngf * 8]       =>  [batch, 16, 16, ngf * 8]
                (self.ngf * 4, 0.0),  # decoder_4: [batch, 16, 16, ngf * 8]     =>  [batch, 32, 32, ngf * 8]
                (self.ngf * 2, 0.0),  # decoder_3: [batch, 32, 32, ngf * 4]     =>  [batch, 64, 64, ngf * 2]
                (self.ngf * 1, 0.0),  # decoder_2: [batch, 64, 64, ngf * 2]     =>  [batch, 128, 128, ngf * 1]
                (self.channels, 0.0),  # decoder_1: [batch, 128, 128, ngf * 1]   =>  [batch, 256, 256, out_channels]
            ]
            decoder = []
            n = len(encoder)
            for (filter, prob) in config:
                m = len(decoder)
                input = tf.concat([decoder[-1], encoder[n - m - 1]], axis=3) if m > 0 else encoder[-1]
                use_bn = False if m == len(config) - 1 else True
                activation = tf.nn.tanh if m == len(config) - 1 else tf.nn.relu
                decoder.append(self.build_decoder_layers(input, filter, prob, use_bn, activation, 'decoder_%d' % (m + 1)))
        return decoder

    def build_unet(self, inputs, reuse=False, name='generator'):
        with tf.variable_scope(name, reuse=reuse):
            encoder = self.build_encoder(inputs)
            decoder = self.build_decoder(encoder)
        return decoder[-1]
