"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import random
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms


def prepare_data(image_paths, bbs, is_flip, opts):
    input_images = [get_image(image_paths[i], opts.image_size, opts.output_size, opts.is_crop, is_flip) for
                          i in range(opts.batch_size)]
    p1xywh = bbs[np.arange(opts.batch_size), 0]
    p2xywh = bbs[np.arange(opts.batch_size), 1]
    p3xywh = bbs[np.arange(opts.batch_size), 2]
    gt_masks = [set_mask(p1xywh, p2xywh, p3xywh, i, opts.output_size) for i in range(opts.batch_size)]
    part1_images = [get_part_image(input_images[i], p1xywh[i], output_size=opts.output_size)
                          for i in range(opts.batch_size)]
    part2_images = [get_part_image(input_images[i], p2xywh[i], output_size=opts.output_size)
                          for i in range(opts.batch_size)]
    part3_images = [get_part_image(input_images[i], p3xywh[i], output_size=opts.output_size)
                          for i in range(opts.batch_size)]
    z = torch.rand([opts.batch_size, opts.z_dim, 1, 1]) * 2.0 - 1.0

    return input_images, part1_images, part2_images, part3_images, gt_masks, z

def set_mask(p1,p2,p3,i, output_size):
    # mask = torch.zeros(3, output_size, output_size)
    # mask[:, p1[i,1]:p1[i,1] + p1[i,3], p1[i,0]:p1[i,0] + p1[i,2]] = 1
    # mask[:, p2[i,1]:p2[i,1] + p2[i,3], p2[i,0]:p2[i,0] + p2[i,2]] = 1
    # mask[:, p3[i,1]:p3[i,1] + p3[i,3], p3[i,0]:p3[i,0] + p3[i,2]] = 1
    mask = torch.zeros(output_size, output_size)
    mask[p1[i,1]:p1[i,1] + p1[i,3], p1[i,0]:p1[i,0] + p1[i,2]] = 1
    mask[p2[i,1]:p2[i,1] + p2[i,3], p2[i,0]:p2[i,0] + p2[i,2]] = 1
    mask[p3[i,1]:p3[i,1] + p3[i,3], p3[i,0]:p3[i,0] + p3[i,2]] = 1


    return mask

def get_image(image_path, image_size, output_size, is_crop, is_flip):
    img = Image.open(image_path).convert('RGB')
    if is_crop:
        # img = center_crop(img, image_size, resize_w=resize_w)
        cx1 = 0.5 * img.size[0] - 0.5 * image_size
        cy1 = 0.5 * img.size[1] - 0.5 * image_size
        cx2 = cx1 + image_size
        cy2 = cy1 + image_size
        img = img.crop([cx1,cy1,cx2,cy2])
    img = img.resize([output_size, output_size], Image.BICUBIC)
    if is_flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    return img

def get_part_image(image, pxywh, output_size):
    i = pxywh[0]
    j = pxywh[1]
    w = pxywh[2]
    h = pxywh[3]
    p_image = image.crop([i,j,i+w,j+h])
    p_image = p_image.resize([output_size, output_size], Image.BICUBIC)

    return p_image


def part_crop(x, pxywh, resize_w=64):
    i = pxywh[0]
    j = pxywh[1]
    w = pxywh[2]
    h = pxywh[3]

    return scipy.misc.imresize(x[j:j+h, i:i+w], [resize_w, resize_w])

def transform(image, isFlip, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    if isFlip:
        cropped_image = np.fliplr(cropped_image)

    return np.array(cropped_image)/127.5 - 1.

def part_transform(image, pxywh, resize_w=64):
    cropped_image = part_crop(image, pxywh, resize_w=resize_w)
    return np.array(cropped_image)/127.5 - 1.


