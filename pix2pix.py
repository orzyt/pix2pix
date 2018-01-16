import numpy as np
import tensorflow as tf
from discriminator import Discriminator
from generator import Generator


class Pix2pix(object):
    def __init__(self, args=None):
        self.args = args
        self.is_training = (args.mode == 'train')
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.args.log_path)

    def build_model(self):
        with tf.name_scope('inputs'):
            self.g_inputs = tf.placeholder(tf.float32, [None, self.args.input_height, self.args.input_width,
                                                        self.args.input_channels], name='g_inputs')
            self.d_inputs = tf.placeholder(tf.float32, [None, self.args.input_height, self.args.input_width,
                                                        self.args.input_channels], name='d_inputs_a')
            self.d_targets = tf.placeholder(tf.float32, [None, self.args.input_height, self.args.input_width,
                                                         self.args.input_channels], name='d_inputs_b')

        with tf.name_scope('discriminator_real'):
            self.d_inputs_real = tf.concat([self.d_inputs, self.d_targets], axis=3)
            self.d_real = Discriminator(self.d_inputs_real, self.is_training, self.args.ndf, reuse=False)

        with tf.name_scope('generator'):
            self.g = Generator(self.g_inputs, self.is_training, self.args.ngf, self.args.out_channels, reuse=False)

        with tf.name_scope('discriminator_fake'):
            self.d_input_fake = tf.concat([self.d_inputs, self.g.outputs], axis=3)
            self.d_fake = Discriminator(self.d_input_fake, self.is_training, self.args.ndf, reuse=True)

        with tf.name_scope('discriminator_loss'):
            self.d_loss = tf.reduce_mean(-(tf.log(self.d_real.logits) + tf.log(1 - self.d_fake.logits)))

        with tf.name_scope('generator_loss'):
            self.g_loss_gan = tf.reduce_mean(-tf.log(self.d_fake.logits))
            self.g_loss_l1 = tf.reduce_mean(tf.abs(self.d_targets - self.g.outputs))
            self.g_loss = self.g_loss_gan + self.args.lam * self.g_loss_l1

        with tf.name_scope('discriminator_train'):
            d_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')
            d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
            with tf.control_dependencies([d_update_ops]):
                self.d_train = tf.train.AdamOptimizer(learning_rate=self.args.lr, beta1=self.args.beta1,
                                                      beta2=self.args.beta2).minimize(self.d_loss, var_list=d_vars)

        with tf.name_scope('generator_train'):
            g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
            g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            with tf.control_dependencies([g_update_ops]):
                self.g_train = tf.train.AdamOptimizer(learning_rate=self.args.lr, beta1=self.args.beta1,
                                                      beta2=self.args.beta2).minimize(self.g_loss, var_list=g_vars)

    def summary(self):
        self.l1_loss_summary = tf.summary.scalar('l1_loss', self.g_loss_l1)
        self.d_loss_summary = tf.summary.scalar('d_loss', self.d_loss)
        self.g_loss_summary = tf.summary.scalar('g_loss', self.g_loss)

        converted_inputs = tf.image.convert_image_dtype(self.d_inputs, dtype=tf.uint8, saturate=True)
        self.inputs_summary = tf.summary.image('inputs', converted_inputs)
        converted_targets = tf.image.convert_image_dtype(self.d_targets, dtype=tf.uint8, saturate=True)
        self.targets_summary = tf.summary.image('targets', converted_targets)
        converted_outputs = tf.image.convert_image_dtype(self.g.outputs, dtype=tf.uint8, saturate=True)
        self.outputs_summary = tf.summary.image('outputs', converted_outputs)

        self.summaries = tf.summary.merge_all()

    def train(self):
        pass