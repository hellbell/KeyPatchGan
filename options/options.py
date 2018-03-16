import argparse
import os
import numpy as np

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # For celebA, is_crop=True, image_size=128, edge_box_resol=128
        # For compcar (128), is_crop=False, edge_box_resol=128
        # For compcar (256), is_crop=False, edge_box_resol=256

        ### TRAINING OPTIONS ###
        self.parser.add_argument('--epoch',         type=int, default=25)
        self.parser.add_argument('--learning_rate', type=float, default=0.0002)
        self.parser.add_argument('--beta1',         type=float, default=0.5)
        self.parser.add_argument('--batch_size',    type=int, default=64)
        self.parser.add_argument('--image_size',    type=int, default=108)
        self.parser.add_argument('--output_size',   type=int, default=64)
        self.parser.add_argument('--c_dim',         type=int, default=3)
        self.parser.add_argument('--conv_dim',        type=int, default=64)
        self.parser.add_argument('--z_dim',         type=int, default=128)
        self.parser.add_argument('--part_embed_dim', type=int, default=128)
        self.parser.add_argument('--edge_box_resol', type=int, default=128)
        self.parser.add_argument('--num_conv_layers', type=int, default=0)
        self.parser.add_argument('--num_train_imgs', type=int, default=np.inf)
        self.parser.add_argument('--is_train',      default=True)
        self.parser.add_argument('--is_crop',       default=True)
        self.parser.add_argument('--cont_train', default=False)
        self.parser.add_argument('--start_epoch', default=0)
        self.parser.add_argument('--model_structure', default='resblock')
        self.parser.add_argument('--res_n_repeat', type=int, default=4)
        self.parser.add_argument('--res_n_downsample', type=int, default=3)
        self.parser.add_argument('--res_n_upsample', type=int, default=3)
        # self.parser.add_argument('--model_structure', default='unet')


        ### OTHER OPTIONS ###
        self.parser.add_argument('--use_gpu', default=True)
        self.parser.add_argument('--use_multigpu', default=True)
        self.parser.add_argument('--gpu_id', default=0)
        self.parser.add_argument('--use_visdom', default=True)
        self.parser.add_argument('--visdom_port', type=int, default=8097)
        self.parser.add_argument('--use_tensorboard', default=True)
        self.parser.add_argument('--tb_log_path', default='logs')
        self.parser.add_argument('--random_seed', default=1004)

        self.parser.add_argument('--num_tests',     type=int, default=128)
        self.parser.add_argument('--num_samples',   type=int, default=128)
        self.parser.add_argument('--sample_dir',    default='results/samples')
        self.parser.add_argument('--test_dir',      default='results/test')
        self.parser.add_argument('--net_dir',      default='nets')


        ### DATABASE OPTIONS ###
        self.parser.add_argument('--db_name', default='celebA')
        # self.parser.add_argument('--db_name', default='compcar_256')
        self.parser.add_argument('--dataset_root', default='/home/sangdoo/work/dataset')

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        self.opt.gpu_id = np.int(self.opt.gpu_id)
        for k,v in sorted(args.items()):
            print('%s: %s' %(str(k), str(v)))
        return self.opt
