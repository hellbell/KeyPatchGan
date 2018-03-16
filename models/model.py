import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from collections import OrderedDict
import numpy as np
import itertools
from PIL import Image
import os
import time
from .networks import PartEncoderR, DiscriminatorR, MaskGeneratorR, ImageGeneratorR
from .networks import PartEncoderU, DiscriminatorU, MaskGeneratorU, ImageGeneratorU
from utils.my_utils import weights_init



class KeyPatchGanModel():
    def __init__(self):
        self.opts = []

    def initialize(self, opts):
        self.opts = opts
        self.batch_size  = self.opts.batch_size
        self.c_dim       = self.opts.c_dim
        self.output_size = self.opts.output_size
        self.z_dim       = self.opts.z_dim

        save_dir_str = str(opts.model_structure) + 'o' + str(opts.output_size) + '_b' + str(opts.batch_size) + \
                        '_df' + str(opts.conv_dim) + '_epch' + str(opts.epoch)
        self.sample_dir = os.path.join(opts.sample_dir, opts.db_name, save_dir_str)
        self.test_dir = os.path.join(opts.test_dir, opts.db_name, save_dir_str)
        self.net_save_dir = os.path.join(opts.net_dir, opts.db_name, save_dir_str)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        if not os.path.exists(self.net_save_dir):
            os.makedirs(self.net_save_dir)

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

        # define networks
        if self.opts.model_structure == 'resblock':
            self.net_part_encoder   = PartEncoderR(self.opts,
                                                   repeat_num=self.opts.res_n_repeat,
                                                   num_downsample=self.opts.res_n_downsample)
            self.net_mask_generator = MaskGeneratorR(self.opts,
                                                     num_upsample=self.opts.res_n_upsample)
            self.net_generator      = ImageGeneratorR(self.opts,
                                                      num_upsample=self.opts.res_n_upsample)
            self.net_discriminator  = DiscriminatorR(self.opts,
                                                     repeat_num=self.opts.res_n_repeat)
        else:
            # find depth of network
            num_conv_layers = 0
            osize = self.opts.output_size / 4
            while (True):
                osize = osize / 2
                if osize < 1:
                    break
                num_conv_layers = num_conv_layers + 1
            self.opts.num_conv_layers = num_conv_layers
            self.net_discriminator = DiscriminatorU(self.opts)
            self.net_generator = ImageGeneratorU(self.opts)
            self.net_part_encoder = PartEncoderU(self.opts)
            self.net_mask_generator = MaskGeneratorU(self.opts)
            # self.net_discriminator.apply(weights_init)
            # self.net_generator.apply(weights_init)
            # self.net_part_encoder.apply(weights_init)
            # self.net_mask_generator.apply(weights_init)

        if self.opts.cont_train:
            self.load(self.opts.start_epoch)

        if self.opts.use_gpu:
            if self.opts.use_multigpu:
                self.net_discriminator = nn.DataParallel(self.net_discriminator).cuda()
                self.net_generator = nn.DataParallel(self.net_generator).cuda()
                self.net_part_encoder = nn.DataParallel(self.net_part_encoder).cuda()
                self.net_mask_generator = nn.DataParallel(self.net_mask_generator).cuda()
            else:
                torch.cuda.set_device(self.opts.gpu_id)

        # define optimizer
        self.criterionMask = torch.nn.L1Loss()
        self.criterionAppr = torch.nn.L1Loss()
        self.criterionGAN = torch.nn.BCEWithLogitsLoss()

        self.optimizer_G = torch.optim.Adam(itertools.chain(self.net_generator.parameters(),
                                                            self.net_part_encoder.parameters(),
                                                            self.net_mask_generator.parameters()),
                                            lr=self.opts.learning_rate,
                                            betas=(self.opts.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.net_discriminator.parameters(),
                                            lr=self.opts.learning_rate,
                                            betas=(self.opts.beta1, 0.999))

        if self.opts.use_tensorboard:
            from utils.logger import Logger
            self.logger = Logger(self.opts.tb_log_path)

        if self.opts.use_visdom:
            import visdom
            self.vis = visdom.Visdom(port=self.opts.visdom_port)






    def forward(self):

        if self.opts.model_structure == 'resblock':
            ''' Encoding Key parts '''
            self.part_enc1 = self.net_part_encoder(self.input_part1)
            self.part_enc2 = self.net_part_encoder(self.input_part2)
            self.part_enc3 = self.net_part_encoder(self.input_part3)
            self.parts_enc = self.part_enc1 + self.part_enc2 + self.part_enc3
            ''' Generating mask'''
            self.gen_mask = self.net_mask_generator(self.parts_enc)
            ''' Generating Full image'''
            self.image_gen = self.net_generator(self.parts_enc)
        else:
            # that means, 'U-net' structure
            ''' Encoding Key parts '''
            self.part1_enc_out = self.net_part_encoder(self.input_part1)
            self.part2_enc_out = self.net_part_encoder(self.input_part2)
            self.part3_enc_out = self.net_part_encoder(self.input_part3)
            self.parts_enc = []
            for val in range(len(self.part1_enc_out)):
                self.parts_enc.append(self.part1_enc_out[val] + self.part2_enc_out[val] + self.part3_enc_out[val])

            ''' Generating mask'''
            self.gen_mask_output = self.net_mask_generator(self.parts_enc)
            self.gen_mask = self.gen_mask_output[-1]

            ''' Generating Full image'''
            self.image_gen_output = self.net_generator(self.parts_enc[-1], self.input_z,
                                                       self.gen_mask_output)
            self.image_gen = self.image_gen_output[-1]



    def backward_D(self):
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
        true_tensor = true_tensor.cuda()
        fake_tensor = Variable(self.Tensor(self.d_real.data.size()).fill_(0.0))
        fake_tensor = fake_tensor.cuda()

        d_loss_real = self.criterionGAN(self.d_real, true_tensor)
        d_loss_fake = self.criterionGAN(self.d_gen, fake_tensor)
        d_loss_shfpart_realbg = self.criterionGAN(self.d_shfpart_realbg, fake_tensor)
        d_loss_realpart_shfbg = self.criterionGAN(self.d_realpart_shfbg, fake_tensor)

        self.d_loss = d_loss_real + d_loss_fake + \
                      d_loss_shfpart_realbg + d_loss_realpart_shfbg
        self.d_loss.backward()

        self.loss['D/loss_all'] = self.d_loss.data[0]
        self.loss['D/loss_real'] = d_loss_real.data[0]
        self.loss['D/loss_fake'] = d_loss_fake.data[0]
        self.loss['D/loss_shfpart_realbg'] = d_loss_shfpart_realbg.data[0]
        self.loss['D/loss_realpart_shfbg'] = d_loss_realpart_shfbg.data[0]


    def backward_G(self):
        self.d_real = self.net_discriminator(self.input_image)
        self.d_gen = self.net_discriminator(self.image_gen)
        self.real_gtpart = torch.mul(self.input_image, self.gt_mask)  # realpart
        self.gen_genpart = torch.mul(self.image_gen, self.gen_mask)  # genpart

        true_tensor = Variable(self.Tensor(self.d_real.size()).fill_(1.0))
        true_tensor = true_tensor.cuda()

        self.g_loss_l1_mask = self.criterionMask(self.gen_mask, self.gt_mask) * self.weight_mask_loss
        self.g_loss_l1_appr = self.criterionAppr(self.gen_genpart, self.real_gtpart) * self.weight_appr_loss
        self.g_loss_gan = self.criterionGAN(self.d_gen, true_tensor)
        self.g_loss = self.g_loss_l1_mask + self.g_loss_l1_appr + self.g_loss_gan
        self.g_loss.backward()

        self.loss['G/loss_all'] = self.g_loss.data[0]
        self.loss['G/loss_fake'] = self.g_loss_gan.data[0]
        self.loss['G/loss_mask'] = self.g_loss_l1_mask.data[0]
        self.loss['G/loss_appr'] = self.g_loss_l1_appr.data[0]





    def optimize_parameters_D(self):
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def optimize_parameters_G(self):
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def visualize(self, win_offset=0):

        # show input image
        # show gen image
        # show pred mask
        input_image = (self.input_image[0:8].cpu().data + 1.0) / 2.0
        image_gen   = (self.image_gen[0:8].cpu().data + 1.0) / 2.0
        gen_mask    = (self.gen_mask[0:8].cpu().data)
        gen_mask = gen_mask * image_gen
        gt_mask     = (self.gt_mask[0:8].cpu().data)
        gt_mask = gt_mask * input_image
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
        gen_mask    = gen_mask * image_gen
        gt_mask     = (self.gt_mask[0:num_img_cols].cpu().data)
        gt_mask     = gt_mask * input_image

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
                   z, gt_mask, weight_g_loss1, weight_g_loss2):

        self.input_image   = Variable(self.Tensor(self.batch_size, self.c_dim, self.output_size, self.output_size))
        self.shuff_image   = Variable(self.Tensor(self.batch_size, self.c_dim, self.output_size, self.output_size))
        self.input_part1   = Variable(self.Tensor(self.batch_size, self.c_dim, self.output_size, self.output_size))
        self.input_part2   = Variable(self.Tensor(self.batch_size, self.c_dim, self.output_size, self.output_size))
        self.input_part3   = Variable(self.Tensor(self.batch_size, self.c_dim, self.output_size, self.output_size))
        self.gt_mask       = Variable(self.Tensor(self.batch_size, 1, self.output_size, self.output_size))
        self.input_z       = Variable(z)
        self.weight_mask_loss = Variable(self.Tensor([weight_g_loss1]))
        self.weight_appr_loss = Variable(self.Tensor([weight_g_loss2]))

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
            self.weight_mask_loss = self.weight_mask_loss.cuda()
            self.weight_appr_loss = self.weight_appr_loss.cuda()

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


