import os
import tensorflow as tf
import time
from glob import glob
from utils import *
from discriminator import Discriminator
from generator import Generator

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class Pix2pix(object):
    def __init__(self, args=None):
        self.args = args
        self.model_name = 'Pix2pix'
        self.is_training = True
        self.data_files = glob(os.path.join(self.args.dataset_dir, self.args.dataset_name, self.args.phase, '*.*'))
        self.data_nums = len(self.data_files)
        self.counter = 0
        self.prefix = self.args.prefix if self.args.prefix != 'None' else get_prefix()
        self.log_file = open(os.path.join(self.prefix, self.args.log_file), 'a')
        self.batches = self.make_batches()

    def build(self):
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.float32, [None, self.args.input_height, self.args.input_width,
                                                      self.args.input_channels], name='inputs')
            self.targets = tf.placeholder(tf.float32, [None, self.args.input_height, self.args.input_width,
                                                       self.args.input_channels], name='targets')

        with tf.name_scope('discriminator'):
            self.d_inputs_real = tf.concat([self.inputs, self.targets], axis=3)
            self.d_real = Discriminator(self.d_inputs_real, self.is_training, self.args.ndf, reuse=False)

        with tf.name_scope('generator'):
            self.g = Generator(self.inputs, self.is_training, self.args.ngf, self.args.out_channels, reuse=False)

        with tf.name_scope('discriminator'):
            self.d_input_fake = tf.concat([self.inputs, self.g.outputs], axis=3)
            self.d_fake = Discriminator(self.d_input_fake, self.is_training, self.args.ndf, reuse=True)

        with tf.name_scope('discriminator_loss'):
            self.d_loss = tf.reduce_mean(-(tf.log(self.d_real.logits + 1e-12) + tf.log(1 - self.d_fake.logits + 1e-12)))

        with tf.name_scope('generator_loss'):
            self.g_loss_gan = tf.reduce_mean(-tf.log(self.d_fake.logits + 1e-12))
            self.g_loss_l1 = tf.reduce_mean(tf.abs(self.targets - self.g.outputs))
            self.g_loss = self.g_loss_gan + self.args.lam * self.g_loss_l1

        with tf.name_scope('discriminator_train'):
            d_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')
            d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
            with tf.control_dependencies(d_update_ops):
                self.d_train = tf.train.AdamOptimizer(learning_rate=self.args.lr, beta1=self.args.beta1,
                                                      beta2=self.args.beta2).minimize(self.d_loss, var_list=d_vars)

        with tf.name_scope('generator_train'):
            g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
            g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            with tf.control_dependencies(g_update_ops):
                self.g_train = tf.train.AdamOptimizer(learning_rate=self.args.lr, beta1=self.args.beta1,
                                                      beta2=self.args.beta2).minimize(self.g_loss, var_list=g_vars)
        with tf.name_scope('ema'):
            self.ema = tf.train.ExponentialMovingAverage(decay=0.99)
            self.d_loss_ema = self.ema.apply([self.d_loss, ])
            self.g_loss_ema = self.ema.apply([self.g_loss, self.g_loss_gan, self.g_loss_l1])

        with tf.name_scope("parameter_count"):
            self.parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

        self.saver = tf.train.Saver(max_to_keep=1)
        self.summary_writer = tf.summary.FileWriter(os.path.join(self.prefix, self.args.summaries_path))

    def summary(self):
        self.gan_loss_summary = tf.summary.scalar('gan_loss', self.ema.average(self.g_loss_gan))
        self.l1_loss_summary = tf.summary.scalar('l1_loss', self.ema.average(self.g_loss_l1))
        self.d_loss_summary = tf.summary.scalar('d_loss', self.ema.average(self.d_loss))
        self.g_loss_summary = tf.summary.scalar('g_loss', self.ema.average(self.g_loss))

        self.inputs_summary = tf.summary.image('inputs', self.inputs)
        self.targets_summary = tf.summary.image('targets', self.targets)
        self.outputs_summary = tf.summary.image('outputs', self.g.outputs)

        self.summaries = tf.summary.merge_all()

    def get_batch(self, data_files):
        data_batch = [[], []]
        for data_file in data_files:
            images = get_image(data_file)
            for i in range(2):
                data_batch[i].append(images[i])
        data_batch = np.array(data_batch).astype(np.float32)
        return data_batch

    def make_batches(self):
        batches = []
        i = 0
        while i + self.args.batch_size <= self.data_nums:
            data_batch = self.get_batch(self.data_files[i: i + self.args.batch_size])
            i += self.args.batch_size
            batches.append(data_batch)
        return np.array(batches)

    def save_model(self, sess, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def test(self, sess):
        ckpt = tf.train.latest_checkpoint(os.path.join(self.prefix, self.args.checkpoint_dir))
        self.saver.restore(sess, ckpt)

        test_dir = os.path.join(self.prefix, self.args.test_dir)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        for batch in self.batches:
            input_A, input_B = batch[0], batch[1]
            outputs = sess.run(self.g.outputs, feed_dict={self.inputs: input_B})
            batch_samples = np.concatenate([input_B, input_A, outputs], axis=2)
            for i in range(self.args.batch_size):
                sample = batch_samples[i, :, :, :]
                sample = (sample + 1.0) * 127.5
                sample = sample.astype('uint8')
                save_images(sample, '{}/{}.png'.format(test_dir, self.counter))
                self.counter += 1
                print('[*] Saved image{}.png'.format(self.counter))

    def train(self, sess):
        self.summary()

        sess.run(tf.global_variables_initializer())
        self.summary_writer.add_graph(tf.get_default_graph())

        log(self.log_file, '[Parameter count]: {}'.format(sess.run(self.parameter_count)))
        start_time = time.time()

        for epoch in range(self.args.epochs):
            log(self.log_file, 'Epoch %d:' % epoch)
            epoch_start_time = time.time()
            np.random.shuffle(self.batches)
            for batch in self.batches:
                input_A, input_B = batch[0], batch[1]
                sess.run([self.d_train, self.d_loss_ema],
                         feed_dict={self.inputs: input_B, self.targets: input_A})
                sess.run([self.g_train, self.g_loss_ema],
                         feed_dict={self.inputs: input_B, self.targets: input_A})
                d_loss, g_loss, l1_loss = sess.run([self.ema.average(self.d_loss),
                                                    self.ema.average(self.g_loss),
                                                    self.ema.average(self.g_loss_l1)])
                if self.counter % self.args.save_summaries == 0:
                    summaries = sess.run(self.summaries, feed_dict={self.inputs: input_B, self.targets: input_A})
                    self.summary_writer.add_summary(summaries, global_step=self.counter)
                if self.counter % self.args.show_loss == 0:
                    log(self.log_file, '[Loss ]: D = %f | G = %f | L1 = %f' % (d_loss, g_loss, l1_loss))

                self.counter += 1

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            images_per_sec = self.data_nums / epoch_duration
            sec_per_batch = epoch_duration / self.data_nums
            log(self.log_file, '[Perf ]: %.1f images / sec, %.3f sec / batch' % (images_per_sec, sec_per_batch))

            duration = (epoch_end_time - start_time) / 60
            left = (self.args.epochs - epoch - 1) / ((epoch + 1) / duration)
            log(self.log_file, '[Time ]: use %.1f min, left %.1f min' % (duration, left))

            if epoch % self.args.save_model == self.args.save_model - 1:
                log(self.log_file, 'Saving model...')
                self.save_model(sess, os.path.join(self.prefix, self.args.checkpoint_dir), self.counter)
