import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from collections import OrderedDict
import numpy as np
import itertools
import visdom
from PIL import Image
import os
import time

class KeyPatchGanModel():
    def __init__(self):
        self.opts = []

    def initialize(self, opts):
        self.opts = opts
        self.batch_size  = self.opts.batch_size
        self.c_dim       = self.opts.c_dim
        self.output_size = self.opts.output_size
        self.z_dim       = self.opts.z_dim

        save_dir_str = 'o' + str(opts.output_size) + '_b' + str(opts.batch_size)
        self.sample_dir = os.path.join(opts.sample_dir, opts.db_name, save_dir_str)
        self.test_dir = os.path.join(opts.test_dir, opts.db_name, save_dir_str)
        self.net_save_dir = os.path.join(opts.net_dir, opts.db_name, save_dir_str)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        if not os.path.exists(self.net_save_dir):
            os.makedirs(self.net_save_dir)


        if self.opts.use_gpu:
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.Tensor

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

        # input part1, part2, part3, images, gtMast, z
        self.input_image = Variable(self.Tensor(self.batch_size, self.c_dim, self.output_size, self.output_size))
        self.shuff_image = Variable(self.Tensor(self.batch_size, self.c_dim, self.output_size, self.output_size))
        self.input_part1  = Variable(self.Tensor(self.batch_size, self.c_dim, self.output_size, self.output_size))
        self.input_part2  = Variable(self.Tensor(self.batch_size, self.c_dim, self.output_size, self.output_size))
        self.input_part3  = Variable(self.Tensor(self.batch_size, self.c_dim, self.output_size, self.output_size))
        self.input_z      = Variable(self.Tensor(self.batch_size, self.z_dim, 1, 1))
        self.gt_mask      = Variable(self.Tensor(self.batch_size, 1, self.output_size, self.output_size))
        self.weight_g_loss = Variable(self.Tensor(1))

        # define net_discriminator
        self.net_discriminator  = Discriminator(self.opts)
        # define net_generator
        self.net_generator      = ImageGenerator(self.opts)
        # define net_part_encoder
        self.net_part_encoder   = PartEncoder(self.opts)
        # define net_part_decoder
        self.net_mask_generator = MaskGenerator(self.opts)

        if self.opts.cont_train:
            self.load(self.opts.start_epoch)


        if self.opts.use_gpu:
            torch.cuda.set_device(self.opts.gpu_id)
            self.net_discriminator.cuda()
            self.net_generator.cuda()
            self.net_part_encoder.cuda()
            self.net_mask_generator.cuda()


        # define optimizer

        self.criterionMask = torch.nn.L1Loss()
        self.criterionAppr = torch.nn.L1Loss()
        self.criterionGAN = torch.nn.BCELoss()

        self.optimizer_G = torch.optim.Adam(itertools.chain(self.net_generator.parameters(),
                                                            self.net_part_encoder.parameters(),
                                                            self.net_mask_generator.parameters()),
                                            lr=self.opts.learning_rate,
                                            betas=(self.opts.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.net_discriminator.parameters(),
                                            lr=self.opts.learning_rate,
                                            betas=(self.opts.beta1, 0.999))

        self.vis = visdom.Visdom(port=self.opts.visdom_port)

    def forward(self, is_train):

        ''' Encoding Key parts '''
        _, self.part1_enc_out = self.net_part_encoder(self.input_part1)
        _, self.part2_enc_out = self.net_part_encoder(self.input_part2)
        _, self.part3_enc_out = self.net_part_encoder(self.input_part3)

        self.parts_enc = {
            'embed': self.part1_enc_out['embed'] + self.part2_enc_out['embed'] + self.part3_enc_out['embed'],
            'e0': self.part1_enc_out['e0'] + self.part2_enc_out['e0'] + self.part3_enc_out['e0'],
            'e1': self.part1_enc_out['e1'] + self.part2_enc_out['e1'] + self.part3_enc_out['e1'],
            'e2': self.part1_enc_out['e2'] + self.part2_enc_out['e2'] + self.part3_enc_out['e2'],
            'e3': self.part1_enc_out['e3'] + self.part2_enc_out['e3'] + self.part3_enc_out['e3']
        }

        ''' Generating mask'''
        self.gen_mask, gen_mask_output = self.net_mask_generator(self.parts_enc['embed'],
                                                self.parts_enc['e0'], self.parts_enc['e1'],
                                                self.parts_enc['e2'], self.parts_enc['e3'])

        ''' Generating Full image'''
        self.image_gen, _ = self.net_generator(self.parts_enc['embed'], self.input_z,
                                              gen_mask_output['m0'], gen_mask_output['m1'],
                                              gen_mask_output['m2'], gen_mask_output['m3'])



    def backward_D(self):
        # self.real_gtother = torch.mul(self.input_image, 1 - self.gt_mask)  # realother
        self.genpart_realbg = torch.mul(self.image_gen, self.gt_mask) + \
                              torch.mul(self.input_image, 1 - self.gt_mask)  # GR
        self.realpart_genbg = torch.mul(self.image_gen, 1 - self.gt_mask) + \
                              torch.mul(self.input_image, self.gt_mask)  # RG
        self.shfpart_realbg = torch.mul(self.shuff_image, self.gt_mask) + \
                              torch.mul(self.input_image, 1 - self.gt_mask)  # SR
        self.realpart_shfbg = torch.mul(self.input_image, self.gt_mask) + \
                              torch.mul(self.shuff_image, 1 - self.gt_mask)  # RS

        self.d_real = self.net_discriminator(self.input_image.detach())
        self.d_gen = self.net_discriminator(self.image_gen.detach())
        self.d_genpart_realbg = self.net_discriminator(self.genpart_realbg.detach())
        self.d_realpart_genbg = self.net_discriminator(self.realpart_genbg.detach())
        self.d_shfpart_realbg = self.net_discriminator(self.shfpart_realbg.detach())
        self.d_realpart_shfbg = self.net_discriminator(self.realpart_shfbg.detach())

        true_tensor = Variable(self.Tensor(self.d_real.data.size()).fill_(1.0))
        fake_tensor = Variable(self.Tensor(self.d_real.data.size()).fill_(0.0))
        self.d_loss = self.criterionGAN(self.d_real, true_tensor) + self.criterionGAN(self.d_gen, fake_tensor) + \
                      self.criterionGAN(self.d_genpart_realbg, fake_tensor) + self.criterionGAN(self.d_realpart_genbg, fake_tensor) + \
                      self.criterionGAN(self.d_shfpart_realbg, fake_tensor) + self.criterionGAN(self.d_realpart_shfbg, fake_tensor)
        self.d_loss.backward()

    def backward_G(self):
        self.d_real = self.net_discriminator(self.input_image)
        self.d_gen = self.net_discriminator(self.image_gen)
        self.real_gtpart = torch.mul(self.input_image, self.gt_mask)  # realpart
        self.gen_genpart = torch.mul(self.image_gen, self.gen_mask)  # genpart

        true_tensor = Variable(self.Tensor(self.d_real.size()).fill_(1.0))
        self.g_loss_l1_appr = self.criterionMask(self.gen_mask, self.gt_mask)
        self.g_loss_l1_mask = self.criterionAppr(self.gen_genpart, self.real_gtpart)
        self.g_loss_gan = self.criterionGAN(self.d_gen, true_tensor)
        self.g_loss = 3.0 * self.weight_g_loss * self.g_loss_l1_appr + self.weight_g_loss * self.g_loss_l1_mask + 2.0 * self.g_loss_gan

        # tt = time.time()
        self.g_loss.backward()
        # print ('%f' % (time.time()-tt))

    def optimize_parameters(self):

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # self.optimizer_G.zero_grad()
        # self.backward_G()
        # self.optimizer_G.step()

    def visualize(self, win_offset=0):

        # show input image
        # show gen image
        # show pred mask
        ups = nn.Upsample(scale_factor=2, mode='nearest')
        input_image = (ups(self.input_image)[0:8].cpu().data + 1.0) / 2.0
        image_gen   = (ups(self.image_gen)[0:8].cpu().data + 1.0) / 2.0
        gen_mask    = (ups(self.gen_mask)[0:8].cpu().data)
        gt_mask     = (ups(self.gt_mask)[0:8].cpu().data)
        self.vis.images(input_image, win=win_offset+0, opts=dict(title='input images'))
        self.vis.images(image_gen,   win=win_offset+1, opts=dict(title='generated images'))
        self.vis.images(gen_mask,    win=win_offset+2, opts=dict(title='predicted masks'))
        self.vis.images(gt_mask,     win=win_offset+3, opts=dict(title='gt masks'))

    def save_images(self, epoch, iter, is_test=False):
        num_img_rows = 7
        num_img_cols = 16

        input_image = (self.input_image[0:num_img_cols].cpu().data + 1.0) / 2.0
        input_part1 = (self.input_part1[0:num_img_cols].cpu().data + 1.0) / 2.0
        input_part2 = (self.input_part2[0:num_img_cols].cpu().data + 1.0) / 2.0
        input_part3 = (self.input_part3[0:num_img_cols].cpu().data + 1.0) / 2.0
        image_gen   = (self.image_gen[0:num_img_cols].cpu().data + 1.0) / 2.0
        gen_mask    = (self.gen_mask[0:num_img_cols].cpu().data)
        gt_mask     = (self.gt_mask[0:num_img_cols].cpu().data)

        input_image_pil = [transforms.ToPILImage()(input_image[i]) for i in range(input_image.shape[0])]
        input_part1_pil = [transforms.ToPILImage()(input_part1[i]) for i in range(input_part1.shape[0])]
        input_part2_pil = [transforms.ToPILImage()(input_part2[i]) for i in range(input_part2.shape[0])]
        input_part3_pil = [transforms.ToPILImage()(input_part3[i]) for i in range(input_part3.shape[0])]
        image_gen_pil   = [transforms.ToPILImage()(image_gen[i])   for i in range(image_gen.shape[0])]
        gen_mask_pil    = [transforms.ToPILImage()(gen_mask[i])    for i in range(gen_mask.shape[0])]
        gt_mask_pil     = [transforms.ToPILImage()(gt_mask[i])     for i in range(gt_mask.shape[0])]

        im_w = input_image.shape[2]
        im_h = input_image.shape[3]

        image_save = Image.new('RGB', (num_img_cols*im_w, num_img_rows*im_h))

        for i in range(num_img_cols):
            image_save.paste(input_part1_pil[i],   (im_w*i, im_h*0))
            image_save.paste(input_part2_pil[i],   (im_w*i, im_h*1))
            image_save.paste(input_part3_pil[i],   (im_w*i, im_h*2))
            image_save.paste(input_image_pil[i],   (im_w*i, im_h*3))
            image_save.paste(image_gen_pil[i],     (im_w*i, im_h*4))
            image_save.paste(gen_mask_pil[i],      (im_w*i, im_h*5))
            image_save.paste(gt_mask_pil[i],       (im_w*i, im_h*6))

        save_name = "epoch_%02d_iter_%04d.png" %(epoch, iter)
        if is_test:
            save_image_path = os.path.join(self.test_dir, save_name)
        else:
            save_image_path = os.path.join(self.sample_dir, save_name)

        image_save.save(save_image_path)


    def set_inputs_for_test(self, input_image, input_part1, input_part2, input_part3, z):


        self.input_image = Variable(self.Tensor(self.batch_size, self.c_dim, self.output_size, self.output_size))
        self.input_part1  = Variable(self.Tensor(self.batch_size, self.c_dim, self.output_size, self.output_size))
        self.input_part2  = Variable(self.Tensor(self.batch_size, self.c_dim, self.output_size, self.output_size))
        self.input_part3  = Variable(self.Tensor(self.batch_size, self.c_dim, self.output_size, self.output_size))
        self.input_z = Variable(z)

        # stack tensors
        for i in range(len(input_image)):
            self.input_image[i,:,:,:] = self.transform(input_image[i])
            self.input_part1[i,:,:,:] = self.transform(input_part1[i])
            self.input_part2[i,:,:,:] = self.transform(input_part2[i])
            self.input_part3[i,:,:,:] = self.transform(input_part3[i])



        if self.opts.use_gpu:
            self.input_image = self.input_image.cuda()
            self.input_part1 = self.input_part1.cuda()
            self.input_part2 = self.input_part2.cuda()
            self.input_part3 = self.input_part3.cuda()
            self.input_z     = self.input_z.cuda()


    def set_inputs_for_train(self, input_image, shuff_image, input_part1, input_part2, input_part3,
                   z, gt_mask, weight_g_loss):

        self.input_image   = Variable(self.Tensor(self.batch_size, self.c_dim, self.output_size, self.output_size))
        self.shuff_image   = Variable(self.Tensor(self.batch_size, self.c_dim, self.output_size, self.output_size))
        self.input_part1   = Variable(self.Tensor(self.batch_size, self.c_dim, self.output_size, self.output_size))
        self.input_part2   = Variable(self.Tensor(self.batch_size, self.c_dim, self.output_size, self.output_size))
        self.input_part3   = Variable(self.Tensor(self.batch_size, self.c_dim, self.output_size, self.output_size))
        self.gt_mask       = Variable(self.Tensor(self.batch_size, 1, self.output_size, self.output_size))
        self.input_z       = Variable(z)
        self.weight_g_loss = Variable(self.Tensor([weight_g_loss]))

        # stack tensors
        for i in range(len(input_image)):
            self.input_image[i,:,:,:] = self.transform(input_image[i])
            self.shuff_image[i,:,:,:] = self.transform(shuff_image[i])
            self.input_part1[i,:,:,:] = self.transform(input_part1[i])
            self.input_part2[i,:,:,:] = self.transform(input_part2[i])
            self.input_part3[i,:,:,:] = self.transform(input_part3[i])
            self.gt_mask[i,0,:,:] = gt_mask[i]

        if self.opts.use_gpu:
            self.input_image = self.input_image.cuda()
            self.shuff_image = self.shuff_image.cuda()
            self.input_part1 = self.input_part1.cuda()
            self.input_part2 = self.input_part2.cuda()
            self.input_part3 = self.input_part3.cuda()
            self.gt_mask    = self.gt_mask.cuda()
            self.input_z    = self.input_z.cuda()
            self.weight_g_loss = self.weight_g_loss.cuda()

    def save(self, epoch):
        self.save_network(self.net_discriminator, epoch, 'net_disc')
        self.save_network(self.net_generator, epoch, 'net_imggen')
        self.save_network(self.net_part_encoder, epoch, 'net_partenc')
        self.save_network(self.net_mask_generator, epoch, 'net_maskgen')

    def load(self, epoch):
        self.load_network(self.net_discriminator, epoch, 'net_disc')
        self.load_network(self.net_generator, epoch, 'net_imggen')
        self.load_network(self.net_part_encoder, epoch, 'net_partenc')
        self.load_network(self.net_mask_generator, epoch, 'net_maskgen')


    def save_network(self, network, epoch, net_name):
        save_filename = 'epoch_%s_net_%s.pth' % (epoch, net_name)
        save_path = os.path.join(self.net_save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if self.opts.use_gpu:
            network.cuda()

    def load_network(self, network, epoch, net_name):
        save_filename = 'epoch_%s_net_%s.pth' % (epoch, net_name)
        save_path = os.path.join(self.net_save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))



class PartEncoder(nn.Module):
    def __init__(self, opts):
        super(PartEncoder, self).__init__()

        self.opts = opts
        self.num_conv_layers = opts.num_conv_layers

        self.conv0 = nn.Sequential(
            nn.Conv2d(self.opts.c_dim, self.opts.df_dim,
                       kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.opts.df_dim, self.opts.df_dim*2,
                       kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(self.opts.df_dim*2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.opts.df_dim*2, self.opts.df_dim*4,
                       kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(self.opts.df_dim*4)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.opts.df_dim*4, self.opts.df_dim*8,
                       kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(self.opts.df_dim*8)
        )
        conv3_spatial_dim = np.int(opts.output_size / np.power(2, self.num_conv_layers))
        self.fc4 = nn.Sequential(
            nn.Conv2d(self.opts.df_dim*8, self.opts.part_embed_dim,
                       kernel_size=conv3_spatial_dim, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(self.opts.part_embed_dim)
        )
        self.model = nn.Sequential(
            self.conv0, self.conv1, self.conv2, self.conv3, self.fc4
        )

    def forward(self, x):
        self.e0 = self.conv0(x)
        self.e1 = self.conv1(self.e0)
        self.e2 = self.conv2(self.e1)
        self.e3 = self.conv3(self.e2)
        self.embed = self.fc4(self.e3)

        outputs= {
            'e0': self.e0,
            'e1': self.e1,
            'e2': self.e2,
            'e3': self.e3,
            'embed': self.embed
        }
        return self.embed, outputs

class MaskGenerator(nn.Module):
    def __init__(self, opts):
        super(MaskGenerator, self).__init__()

        self.opts = opts

        df_dims_in = [opts.part_embed_dim,
                      opts.df_dim * 8 * 2,
                      opts.df_dim * 4 * 2,
                      opts.df_dim * 2 * 2,
                      opts.df_dim * 2]


        df_dims_out = [opts.df_dim * 8,
                       opts.df_dim * 4,
                       opts.df_dim * 2,
                       opts.df_dim,
                       1]


        conv3_spatial_dim = np.int(self.opts.output_size / np.power(2, self.opts.num_conv_layers))
        self.convT0 = nn.Sequential(
            nn.ConvTranspose2d(df_dims_in[0], df_dims_out[0],
                               kernel_size=conv3_spatial_dim, stride=1, padding=0),
            nn.BatchNorm2d(opts.df_dim * 8),
            nn.ReLU()
        )
        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(df_dims_in[1], df_dims_out[1],
                               kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(opts.df_dim * 4)
        )
        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(df_dims_in[2], df_dims_out[2],
                               kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(opts.df_dim * 2)
        )
        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(df_dims_in[3], df_dims_out[3],
                               kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(opts.df_dim)
        )
        self.convT4 = nn.Sequential(
            nn.ConvTranspose2d(df_dims_in[4], df_dims_out[4],
                               kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        self.model = nn.Sequential(
            self.convT0, self.convT1, self.convT2, self.convT3, self.convT4
        )

    def forward(self, embed, e0, e1, e2, e3):
        self.m0 = self.convT0(embed)
        self.m0 = torch.cat([self.m0, e3], 1)
        self.m1 = self.convT1(self.m0)
        self.m1 = torch.cat([self.m1, e2], 1)
        self.m2 = self.convT2(self.m1)
        self.m2 = torch.cat([self.m2, e1], 1)
        self.m3 = self.convT3(self.m2)
        self.m3 = torch.cat([self.m3, e0], 1)
        self.mask = self.convT4(self.m3)
        outputs= {
            'm0': self.m0,
            'm1': self.m1,
            'm2': self.m2,
            'm3': self.m3,
            'mask': self.mask
        }
        return self.mask, outputs


class Discriminator(nn.Module):
    def __init__(self, opts):
        super(Discriminator, self).__init__()

        self.opts = opts
        self.num_conv_layers = opts.num_conv_layers

        self.conv0 = nn.Sequential(
            nn.Conv2d(self.opts.c_dim, self.opts.df_dim,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.opts.df_dim, self.opts.df_dim * 2,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(self.opts.df_dim * 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.opts.df_dim * 2, self.opts.df_dim * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(self.opts.df_dim * 4)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.opts.df_dim * 4, self.opts.df_dim * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(self.opts.df_dim * 8)
        )
        conv3_spatial_dim = np.int(opts.output_size / np.power(2, self.num_conv_layers))
        self.fc4 = nn.Sequential(
            nn.Conv2d(self.opts.df_dim * 8, 1,
                      kernel_size=conv3_spatial_dim, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.model = nn.Sequential(
            self.conv0, self.conv1, self.conv2, self.conv3, self.fc4
        )


    def forward(self, img):
        self.d0 = self.conv0(img)
        self.d1 = self.conv1(self.d0)
        self.d2 = self.conv2(self.d1)
        self.d3 = self.conv3(self.d2)
        self.disc = self.fc4(self.d3)
        return self.disc

class ImageGenerator(nn.Module):
    def __init__(self, opts):
        super(ImageGenerator, self).__init__()

        self.opts = opts


        df_dims_in = [opts.part_embed_dim + opts.z_dim,
                      opts.df_dim * 8 * 3,
                      opts.df_dim * 4 * 3,
                      opts.df_dim * 2 * 3,
                      opts.df_dim * 3]

        df_dims_out = [opts.df_dim * 8,
                       opts.df_dim * 4,
                       opts.df_dim * 2,
                       opts.df_dim,
                       opts.c_dim]


        conv3_spatial_dim = np.int(self.opts.output_size / np.power(2, self.opts.num_conv_layers))
        self.convT0 = nn.Sequential(
            nn.ConvTranspose2d(df_dims_in[0], df_dims_out[0],
                               kernel_size=conv3_spatial_dim, stride=1, padding=0),
            nn.BatchNorm2d(opts.df_dim * 8),
            nn.ReLU()
        )
        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(df_dims_in[1], df_dims_out[1],
                               kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(opts.df_dim * 4)
        )
        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(df_dims_in[2], df_dims_out[2],
                               kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(opts.df_dim * 2)
        )
        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(df_dims_in[3], df_dims_out[3],
                               kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(opts.df_dim)
        )
        self.convT4 = nn.Sequential(
            nn.ConvTranspose2d(df_dims_in[4], df_dims_out[4],
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        self.model = nn.Sequential(
            self.convT0, self.convT1, self.convT2, self.convT3, self.convT4
        )



    def forward(self, embed, z, m0, m1, m2, m3):
        self.embed_z = torch.cat([embed, z], 1)
        self.g0 = self.convT0(self.embed_z)
        self.g0 = torch.cat([self.g0, m0], 1)
        self.g1 = self.convT1(self.g0)
        self.g1 = torch.cat([self.g1, m1], 1)
        self.g2 = self.convT2(self.g1)
        self.g2 = torch.cat([self.g2, m2], 1)
        self.g3 = self.convT3(self.g2)
        self.g3 = torch.cat([self.g3, m3], 1)
        self.gen_image = self.convT4(self.g3)
        outputs = {
            'g0': self.g0,
            'g1': self.g1,
            'g2': self.g2,
            'g3': self.g3,
            'gen_image': self.gen_image
        }

        return self.gen_image, outputs
