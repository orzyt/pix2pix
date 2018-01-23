import tensorflow as tf
import argparse
from pix2pix import Pix2pix

parser = argparse.ArgumentParser()

parser.add_argument('--lam', type=float, default=100, help='weights of l1 loss')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2')

parser.add_argument('--epochs', type=int, default=200, help='training epochs')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--input_width', type=int, default=256, help='input width')
parser.add_argument('--input_height', type=int, default=256, help='input height')
parser.add_argument('--input_channels', type=int, default=3, help='input channels')
parser.add_argument('--out_channels', type=int, default=3, help='out channels')
parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters')
parser.add_argument('--ngf', type=int, default=64, help='number of generator filters')

parser.add_argument('--show_loss', type=int, default=100, help='show loss')
parser.add_argument('--save_summaries', type=int, default=50, help='save summaries')
parser.add_argument('--save_model', type=int, default=10, help='save model')
parser.add_argument('--sample', type=int, default=0, help='sample')

parser.add_argument('--dataset_dir', type=str, default='datasets', help='dataset directory')
parser.add_argument('--dataset_name', type=str, default='facades', help='dataset name')
parser.add_argument('--phase', type=str, default='train', help='indicating training or testing phase')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint4', help='checkpoint directory')
parser.add_argument('--test_dir', type=str, default='test4', help='test directory')
parser.add_argument('--sample_dir', type=str, default='sample', help='sample directory')
parser.add_argument('--summaries_path', type=str, default='summaries4', help='summaries directory')
parser.add_argument('--log_file', type=str, default='log4.txt', help='log file')


def main():
    args = parser.parse_args()

    model = Pix2pix(args)
    model.build()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    with tf.Session(config=config) as sess:
        if args.phase == 'train':
            model.train(sess)
        elif args.phase == 'test':
            model.test(sess)


if __name__ == '__main__':
    main()
