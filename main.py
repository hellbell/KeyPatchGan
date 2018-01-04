import numpy as np
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from glob import glob
import scipy.io
import os
import matplotlib.pyplot as plt
import visdom
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
# Initialize Settings
###############################################################





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

print (model.net_part_encoder)




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
m_weight = np.logspace(-2, -4, num=opts.epoch) * 64*64
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

        # forward run
        model.set_inputs_for_train(train_images, train_shuff_images,
                                   train_part1_images, train_part2_images, train_part3_images,
                                   train_z, train_gt_masks, m_weight[epoch])
        model.forward(is_train=True)
        # backward run
        model.optimize_parameters()

        if (i % 100 == 1):
            print('epoch: %02d/%02d, iter: %04d/%04d, d_loss: %f. g_loss_appr: %f, g_loss_mask: %f, %f sec'
                  % (epoch+1, opts.epoch, i, num_batches, model.d_loss.cpu().data.numpy(),
                     model.g_loss_l1_appr.cpu().data.numpy(), model.g_loss_l1_mask.cpu().data.numpy(),
                     time.time()-start_time))


            model.set_inputs_for_train(sample_images, sample_images,
                                       sample_part1_images, sample_part2_images, sample_part3_images,
                                       sample_z, sample_gt_masks, m_weight[epoch])
            model.forward(is_train=False)
            model.visualize(win_offset=0)
            model.set_inputs_for_train(test_images, test_images,
                                       test_part1_images, test_part2_images, test_part3_images,
                                       test_z, test_gt_masks, m_weight[epoch])
            model.forward(is_train=False)
            model.visualize(win_offset=100)

            start_time = time.time()

        if (i % 200 == 1):
            model.set_inputs_for_train(sample_images, sample_images,
                                       sample_part1_images, sample_part2_images, sample_part3_images,
                                       sample_z, sample_gt_masks, m_weight[epoch])
            model.forward(is_train=False)
            model.save_images(epoch, i, is_test=False)
            model.set_inputs_for_train(test_images, test_images,
                                       test_part1_images, test_part2_images, test_part3_images,
                                       test_z, test_gt_masks, m_weight[epoch])
            model.forward(is_train=False)
            model.save_images(epoch, i, is_test=True)

    model.save(epoch)





#
# img_path, bbox = dataset[100]
# img = get_image(img_path, False, 108, is_crop=True, resize_w=64)
#
#
#
# img = np.uint8((img + 1.0)* 255.0 * 0.5)
# fig, ax = plt.subplots(figsize=(12, 12))
# ax.imshow(img, aspect='equal')
# for i in range(3):
#     bb = bbox[i]
#     ax.add_patch(
#         plt.Rectangle((bb[0], bb[1]),
#         bb[2],
#         bb[3], fill=False,
#         edgecolor='red', linewidth=3.5)
#     )
# plt.axis('off')
# plt.tight_layout()
# plt.draw()
# plt.show()



# input = Variable(torch.randn((64,3,64,64)))
# input_p1 = Variable(torch.randn((64,3,64,64)))
# input_p2 = Variable(torch.randn((64,3,64,64)))
# input_p3 = Variable(torch.randn((64,3,64,64)))
# z = Variable(torch.randn((64,100,1,1)))
# gt_mask = Variable(torch.randn((64,1,64,64)))


# input = Variable(torch.randn((64,3,64,64)))
# enc_outputs = model.net_part_encoder(input)
# mask_outputs = model.net_mask_generator(enc_outputs['embed'],
#                                     enc_outputs['e0'], enc_outputs['e1'],
#                                     enc_outputs['e2'], enc_outputs['e3'])
# z = Variable(torch.randn([64,100,1,1]))
# gen_outputs = model.net_generator(enc_outputs['embed'], z,
#                                   mask_outputs['m0'], mask_outputs['m1'],
#                                   mask_outputs['m2'], mask_outputs['m3'])
# disc_outputs = model.net_discriminator(gen_outputs['gen_image'])
#
#
# print(enc_outputs['e0'].size())
# print(enc_outputs['e1'].size())
# print(enc_outputs['e2'].size())
# print(enc_outputs['e3'].size())
# print(enc_outputs['embed'].size())
#
# print(mask_outputs['m0'].size())
# print(mask_outputs['m1'].size())
# print(mask_outputs['m2'].size())
# print(mask_outputs['m3'].size())
# print(mask_outputs['mask'].size())
#
# print(gen_outputs['g0'].size())
# print(gen_outputs['g1'].size())
# print(gen_outputs['g2'].size())
# print(gen_outputs['g3'].size())
# print(gen_outputs['gen_image'].size())
#
# print(disc_outputs['d0'].size())
# print(disc_outputs['d1'].size())
# print(disc_outputs['d2'].size())
# print(disc_outputs['d3'].size())
# print(disc_outputs['disc'].size())


# # visualize
# img = transforms.ToTensor()(part1_images[0])
# vis.image(img, opts=dict(title='Test Image'), win=3)
# img = transforms.ToTensor()(part2_images[0])
# vis.image(img, opts=dict(title='Test Image'), win=4)
# img = transforms.ToTensor()(part3_images[0])
# vis.image(img, opts=dict(title='Test Image'), win=5)
