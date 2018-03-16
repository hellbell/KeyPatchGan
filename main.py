import numpy as np
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from glob import glob
import scipy.io
import os
import time

from data.database import *
from utils.my_utils import *
from options.options import *
from models.model import KeyPatchGanModel

###############################################################
# Get Options
###############################################################
opts = Options().parse()


###############################################################
# Initialize Database
###############################################################
dataset = Dataset()
dataset.initialize(opts)


# Split train/test data
np.random.seed(opts.random_seed)
all_idx = np.random.permutation(len(dataset))
test_idx = all_idx[-opts.num_tests:]
sample_idx = all_idx[:opts.num_samples]
train_idx = all_idx[:-opts.num_tests]
num_train_imgs = len(train_idx)

###############################################################
# Initialize Model
###############################################################
model = KeyPatchGanModel()
model.initialize(opts)

###############################################################
# Start Training
###############################################################

''' Preparing Test Data '''
# set test images
test_img_paths, test_bbs = dataset[test_idx]
is_flip = False
test_images, test_part1_images, test_part2_images, test_part3_images, test_gt_masks, test_z = \
    prepare_data(test_img_paths, test_bbs, is_flip, opts)

''' Preparing Sample Data '''
# set sample images
sample_img_paths, sample_bbs = dataset[sample_idx]
is_flip = False
sample_images, sample_part1_images, sample_part2_images, sample_part3_images, sample_gt_masks, sample_z = \
    prepare_data(sample_img_paths, sample_bbs, is_flip, opts)


''' Main Training Loop Here '''
# compcar (4, 2) try !
# m_weight1 = np.logspace(-1.5, -3.5, num=opts.epoch)
# m_weight2 = np.logspace(-2, -4, num=opts.epoch)
m_weight_mask = np.logspace(1, 1, num=opts.epoch)
m_weight_appr = np.logspace(1, 1, num=opts.epoch) * 10


start_time = time.time()
for epoch in range(opts.epoch):
    # shuffle data
    curr_epoch_idx = np.random.permutation(num_train_imgs)
    curr_train_idx = train_idx[curr_epoch_idx]
    num_batches = num_train_imgs // opts.batch_size

    for i in range(num_batches):
        batch_idx_offset = i * opts.batch_size
        batch_train_idx = curr_train_idx[batch_idx_offset:batch_idx_offset+opts.batch_size]
        batch_train_other_idx = curr_train_idx[np.setdiff1d(np.arange(len(curr_train_idx)),
                                       np.arange(batch_idx_offset, batch_idx_offset+opts.batch_size))]
        batch_shuff_idx = np.random.choice(batch_train_other_idx, size=opts.batch_size)

        train_image_paths, train_bbs = dataset[batch_train_idx]
        shuff_image_paths, _         = dataset[batch_shuff_idx]

        if np.random.rand() > 0.5:
            is_flip = True
            train_bbs[:, :, 0] = opts.output_size - (train_bbs[:, :, 0] + train_bbs[:, :, 2])
        else:
            is_flip = False

        # load images
        train_images, train_part1_images, train_part2_images, train_part3_images, train_gt_masks, train_z = \
            prepare_data(train_image_paths, train_bbs, is_flip, opts)
        train_shuff_images = [get_image(shuff_image_paths[j], opts.image_size, opts.output_size, opts.is_crop, is_flip) for
                        j in range(opts.batch_size)]

        # Set input images
        model.set_inputs_for_train(train_images, train_shuff_images,
                                   train_part1_images, train_part2_images, train_part3_images,
                                   train_z, train_gt_masks, m_weight_mask[epoch],m_weight_appr[epoch])

        # Train D
        model.forward()
        model.optimize_parameters_D()

        # Train G
        if (i + 1) % 2 == 0:
            model.forward()
            model.optimize_parameters_G()


        if (i % 100 == 1):
            print('epoch: %02d/%02d, iter: %04d/%04d, d_loss: %f. g_loss_gan: %f, g_loss_appr: %f, g_loss_mask: %f, %f sec'
                  % (epoch+1, opts.epoch, i, num_batches, model.d_loss.cpu().data.numpy(),
                     model.g_loss_gan.cpu().data.numpy(),
                     model.g_loss_l1_appr.cpu().data.numpy(),
                     model.g_loss_l1_mask.cpu().data.numpy(),
                     time.time()-start_time))

            if opts.visualize:
                model.set_inputs_for_train(sample_images, sample_images,
                                           sample_part1_images, sample_part2_images, sample_part3_images,
                                           sample_z, sample_gt_masks, m_weight_mask[epoch],m_weight_appr[epoch])
                model.forward()
                model.visualize(win_offset=0)
                model.set_inputs_for_train(test_images, test_images,
                                           test_part1_images, test_part2_images, test_part3_images,
                                           test_z, test_gt_masks, m_weight_mask[epoch],m_weight_appr[epoch])
                model.forward()
                model.visualize(win_offset=100)


        if (i % 200 == 1):
            model.set_inputs_for_train(sample_images, sample_images,
                                       sample_part1_images, sample_part2_images, sample_part3_images,
                                       sample_z, sample_gt_masks, m_weight_mask[epoch],m_weight_appr[epoch])
            model.forward()
            model.save_images(epoch, i, is_test=False)
            model.set_inputs_for_train(test_images, test_images,
                                       test_part1_images, test_part2_images, test_part3_images,
                                       test_z, test_gt_masks, m_weight_mask[epoch],m_weight_appr[epoch])
            model.forward()
            model.save_images(epoch, i, is_test=True)

    model.save(epoch)





