import argparse
import os
from util import util
import torch

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):

        parser.add_argument('--st_root', type=str, default=r'./data/datasets/structure', help='path to structure images')
        parser.add_argument('--de_root', type=str, default=r'./data/datasets/images', help='path to detail images (which are the groundtruth)')
        parser.add_argument('--mask_root', type=str, default=r'./data/datasets/mask', help='path to mask, we use the datasetsets of partial conv hear')
        parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        parser.add_argument('--num_workers', type=int, default=4, help='numbers of the core of CPU')
        parser.add_argument('--name', type=str, default='Mutual Encoder-Decoder',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        parser.add_argument('--input_nc', type=int, default=6, help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2')
        parser.add_argument('--model', type=str, default='training1', help='set the names of current training process')
        parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')

        parser.add_argument('--lambda_L1', type=int, default=1, help='weight on L1 term in objective')
        parser.add_argument('--lambda_S', type=int, default=250, help='weight on Style loss in objective')
        parser.add_argument('--lambda_P', type=int, default=0.2, help='weight on Perceptual loss in objective')
        parser.add_argument('--lambda_Gan', type=int, default=0.2, help='weight on GAN term in objective')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)


        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt

