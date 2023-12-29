import torch
import torch.nn as nn
from torch.nn import init

import functools
from torch.optim import lr_scheduler

from torchvision import transforms
#from keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate
#from keras.models import Model
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
import numpy as np
from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import torch.nn.functional as F

from torchvision import models

#import tensorflow as tf
###############################################################################
# Helper Functions
###############################################################################

class Identity(nn.Module):
    def forward(self, x):
        return x




def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
一个channel内做归一化，算H*W的均值，用在风格化迁移；因为在图像风格化中，生成结果主要依赖于某个图像实例，所以对整个batch归一化不适合图像风格化中，
因而对HW做归一化。可以加速模型收敛，并且保持每个图像实例之间的独立
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    学习率调整
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    # 自定义调整学习率
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    # 等间隔调整学习率

    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    # 自适应调整学习率
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    # 余弦退火调整学习率

    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
初始化网络权重。



参数:

net (network)——要初始化的网络

init_type (str)——初始化方法的名称:normal | xavier | kaim |正交
init_gain (float)——法线、xavier和正交的比例因子。
我们在原始的pix2pix和CycleGAN文件中使用“normal”。但xavier和kaim可能会
在某些应用程序中工作得更好。你可以自己试试。
  在深度学习中，神经网络的权重初始化方法对（weight initialization）对模型的收敛速度和性能有着至关重要的影响。说白了，
  神经网络其实就是对权重参数w的不停迭代更新，以期达到较好的性能。在深度神经网络中，随着层数的增多，我们在梯度下降的过程中，
  极易出现梯度消失或者梯度爆炸。因此，对权重w的初始化则显得至关重要，一个好的权重初始化虽然不能完全解决梯度消失和梯度爆炸的问题，
  但是对于处理这两个问题是有很大的帮助的，并且十分有利于模型性能和收敛速度。在这篇博客中，我们主要讨论四种权重初始化方法：

  kaiming提出了一种针对ReLU的初始化方法，一般称作 He initialization。初始化方式为
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='kaiming', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    print('初始化网络参数的类型：', init_type)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='instance', use_dropout=False, init_type='normal', init_gain=0.02,
             gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    print('生成器的初始化norm', norm)
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'G_A':
        net = ResnetGeneratorA(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'G_B':
        net = ResnetGeneratorA(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator
    我们目前的实现提供了三种类型的鉴别器:
    [basic]:在最初的pix2pix论文中描述的“PatchGAN”分类器。
    可以区分70×70重叠斑块的真假。
    这样的补丁级鉴别器架构具有较少的参数
    比全图像鉴别器和可以工作任意大小的图像
    以完全卷积的方式。
    [n_layers]:在这个模式下，你可以在鉴别器中指定conv层的数量
    使用参数(默认为[basic] (PatchGAN)中使用的3)。
    【pixel】:1x1 PixelGAN鉴别器可以对一个像素进行真假分类。
    它鼓励更大的颜色多样性，但对空间统计没有影响。
    鉴别器已由初始化。对非线性采用漏泄式继电器
    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    print('判别器的初始化模型', norm)
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':  # classify if each pixel is real or fake
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
        #net =Discriminator(input_nc)
       
        #net = MultiscaleDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
        #net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
        #PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
        #net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)



##############################################################################
# Classes

class GANLoss_NOEVO(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss_NOEVO, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()

        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
创建与输入大小相同的标签张量。
参数:
预测(张量)——tpyically从一个鉴别器的预测
target_is_real (bool)——如果ground truth标签用于真实图像或虚假图像

返回:
一个标签张量填满地面真值标签，并与输入的大小
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

class GANLoss(nn.Module):
    """Define different GAN Discriminator's objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, loss_mode, which_net, which_D, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GAN's Discriminator Loss class.

        Parameters:
            loss_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss_mode = loss_mode
        self.which_net = which_net 
        self.which_D = which_D 

        if loss_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif loss_mode in ['vanilla', 'nsgan', 'rsgan']:
            self.loss = nn.BCEWithLogitsLoss()
        elif loss_mode in ['wgan', 'hinge']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % loss_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def G_loss(self, Dfake, Dreal):
        real_tensor = self.get_target_tensor(Dreal, True)
        fake_tensor = self.get_target_tensor(Dreal, False)

        if self.which_D == 'S':
            prediction_fake = Dfake
            prediction_real = real_tensor if self.loss_mode in ['vanilla'] else fake_tensor
        elif self.which_D == 'Ra':
            prediction_fake = Dfake - torch.mean(Dreal)
            prediction_real = Dreal - torch.mean(Dfake)
        else:
            raise NotImplementedError('which_D name [%s] is not recognized' % self.which_D)

        if self.loss_mode in ['lsgan', 'nsgan']:
            loss_fake = self.loss(prediction_fake, real_tensor)
            loss_real = self.loss(prediction_real, fake_tensor)
        elif self.loss_mode == 'vanilla':
            loss_fake = -self.loss(prediction_fake, fake_tensor)
            loss_real = -self.loss(prediction_real, real_tensor)
        elif self.loss_mode in ['wgan', 'hinge'] and self.which_D == 'S':
           
            loss_fake = -prediction_fake.mean()
            loss_real =  prediction_real.mean()
        elif self.loss_mode == 'hinge' and self.which_D == 'Ra':
            loss_fake = nn.ReLU()(1.0 - prediction_fake).mean()
            loss_real = nn.ReLU()(1.0 + prediction_real).mean()
        elif self.loss_mode == 'rsgan':
            loss_fake = self.loss(Dfake - Dreal, real_tensor)
            loss_real = 0. 
        else:
            raise NotImplementedError('loss_mode name [%s] is not recognized' % self.loss_mode)

        return loss_fake, loss_real

    def D_loss(self, Dfake, Dreal):
        real_tensor = self.get_target_tensor(Dreal, True)
        fake_tensor = self.get_target_tensor(Dreal, False)

        if self.which_D == 'S':
            prediction_fake = Dfake
            prediction_real = Dreal 
        elif self.which_D == 'Ra':
            prediction_fake = Dfake - torch.mean(Dreal)
            prediction_real = Dreal - torch.mean(Dfake)
        else:
            raise NotImplementedError('which_D name [%s] is not recognized' % self.which_D)

        if self.loss_mode in ['lsgan', 'nsgan', 'vanilla']:
            loss_fake = self.loss(prediction_fake, fake_tensor)
            loss_real = self.loss(prediction_real, real_tensor)
        elif self.loss_mode == 'wgan':
            loss_fake = prediction_fake.mean()
            loss_real = -prediction_real.mean()
        elif self.loss_mode == 'hinge':
            loss_fake = nn.ReLU()(1.0 + prediction_fake).mean()
            loss_real = nn.ReLU()(1.0 - prediction_real).mean()
        elif self.loss_mode == 'rsgan':
            loss_fake = 0. 
            loss_real = self.loss(Dreal - Dfake, real_tensor)
        else:
            raise NotImplementedError('loss_mode name [%s] is not recognized' % self.loss_mode)

        return loss_fake, loss_real

    def __call__(self, Dfake, Dreal):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.which_net == 'G':
            loss_fake, loss_real = self.G_loss(Dfake, Dreal) 
            return loss_fake, loss_real
        elif self.which_net == 'D':
            loss_fake, loss_real = self.D_loss(Dfake, Dreal) 
            return loss_fake, loss_real
        else:
            raise NotImplementedError('which_net name [%s] is not recognized' % self.which_net)

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out



def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
                *real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None



class ResnetGeneratorA(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=True, n_blocks=6,
                 padding_type='reflect',kernel_size=3):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """

        assert (n_blocks >= 0)
        super(ResnetGeneratorA, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # model = [nn.ConvTranspose2d(ngf , 64,kernel_size=3, stride=2,
        #                                  padding=1, output_padding=1,
        #                                  bias=use_bias),norm_layer(ngf),nn.ReLU(True)]

        
        
        model = [nn.ReflectionPad2d(2),
                 nn.Conv2d(3, 64, kernel_size=3, padding=0, bias=use_bias,dilation=2),
                 norm_layer(ngf),
                 nn.ReLU()]
        
        out_features = int(ngf)
        for i in range(9):  # add ResNet blocks

            model += [ResnetBlock(ngf,kernel=3,padding_type=padding_type, norm_layer=norm_layer, use_dropout=False,
                                  use_bias=use_bias)]
        # self.ca = ChannelAttention(out_features)
        # model +=[self.ca]
        model += [nn.ReflectionPad2d(2),
                 nn.Conv2d(64, 64, kernel_size=3, padding=0, bias=use_bias, dilation=2),
                 norm_layer(ngf),
                 nn.Tanh()]

        self.model = nn.Sequential(*model) #w*h*64



        model2 = [nn.ReflectionPad2d(1),
                 nn.Conv2d(3, 64, kernel_size=3, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU()]

        for i in range(6):  # add ResNet blocks

            model2 += [ResnetBlock(64, kernel=1, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False,
                                   use_bias=use_bias)]

        #self.ca = ChannelAttention(out_features)
        #model2 +=[self.ca]
    
        model2 += [nn.ReflectionPad2d(2),
                   nn.Conv2d(64, 32, kernel_size=3, padding=0, bias=use_bias, dilation=2),
                   norm_layer(ngf),
                   nn.Tanh()
                   ]

        self.model2 = nn.Sequential(*model2)


        model3 = [nn.ReflectionPad2d(1),
         nn.Conv2d(3, 32, kernel_size=3, padding=0, bias=use_bias),
         norm_layer(ngf),
         nn.ReLU()]
       

        for i in range(6):  # add ResNet blocks
                 model3 += [ResnetBlock(32,kernel=5,padding_type='reflect', norm_layer=norm_layer, use_dropout=False,
                          use_bias=use_bias)]
        #self.ca = ChannelAttention(in_features)
        #model3 +=[self.ca]

        model3 += [nn.ReflectionPad2d(1),
         nn.Conv2d(32, 32, kernel_size=3, padding=0, bias=use_bias),
         norm_layer(ngf),
         nn.ReLU()]
        self.layer_X3 = nn.Sequential(*model3[:-5])  # Extracts output before last ResnetBlock
        self.layer_X4 = nn.Sequential(*model3)  # Extracts final output
    



        self.conv7 = nn.Conv2d(192, 192, kernel_size=1)
        
        # Smoothing and Output
        self.conv8 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv9 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.conv2_128 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, dilation=2,padding=2),
            norm_layer(64),
            nn.ReLU()
        )
        self.conv3_64 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=2,padding=2),
            nn.ReLU()
        )
        self.conv5_64 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=5, stride=1, dilation=2, padding=4),
            nn.Tanh()
        )
        self.conv2_192 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, stride=1, dilation=2,padding=2),
            norm_layer(64),
            nn.ReLU()
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, dilation=2,padding=2),
            nn.Tanh()
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            # nn.BatchNorm1d(3),
            nn.Linear(192, 192),
            nn.ReLU(),
            nn.Linear(192, 1),  # 可能应该是nn.Linear(channel, channel)
            nn.Sigmoid() #lp 改
            

            
        )
    def forward(self, input):
        """Standard forward"""



        x1 = self.model(input)#64
        
       
        
        x2 = self.model2(input)#64
        
        x3 = self.layer_X3(input)
        
        x4 = self.layer_X4(input)
        
        Y1 = torch.cat((x3, x4), 1)
        
        Y2 = torch.cat((x2, x3), 1)
        
        Out = torch.cat((x1, Y1, Y2), dim=1)
        #lp = Out
        Out = self.conv7(Out)
        
        # Smoothing and Output
        # x_raw = Out
        # x = Out
        # y = torch.abs(x)
        # y_abs = y
        # y = self.gap(y)
        # y = torch.flatten(y, 1)
        # average = y
        # y = self.fc(y)
        # y = torch.mul(average, y)
        # y = y.unsqueeze(2).unsqueeze(2)
        # sub = y_abs - y
        # zeros = sub - sub
        # n_sub = torch.max(sub, zeros)
        # y = torch.mul(torch.sign(x_raw), n_sub)



        # x3 = self.conv2_128(x3)
        out = self.conv2_192(Out)
        out = self.conv3_64(out)
        out = self.conv3_3(out)
        return out

        # return x1, x2, x3, x4,Y1,Y2,out#修改了这里

        
class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, kernel,padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim,kernel, padding_type, norm_layer, use_dropout, use_bias)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(

            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, 1),  # 可能应该是nn.Linear(channel, channel)
            nn.Sigmoid(),
        )
        # self.conv_block2 = self.build_conv_block(dim,)
    def build_conv_block(self, dim, kernel,padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        if kernel==3:
          p = 1
        else:
          p=0
        # if padding_type == 'reflect':
        #     conv_block += [nn.ReflectionPad2d(1)]
        # elif padding_type == 'replicate':
        #     conv_block += [nn.ReplicationPad2d(1)]
        # elif padding_type == 'zero':
        #     p = 1
        # else:
        #     raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, 128, kernel_size=kernel, padding=p, bias=use_bias), norm_layer(64), nn.ReLU(True)]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        # conv_block += [nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=use_bias), nn.ReLU(True)]
        # conv_block += [nn.Conv2d(64, 128, kernel_size=kernel, padding=p, bias=use_bias), nn.ReLU(True)]


        conv_block += [nn.Conv2d(128, dim, kernel_size=kernel, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""

        x_raw = x
        y = torch.abs(x)
        y_abs = y
        y = self.gap(y)
        y = torch.flatten(y, 1)
        average = y
        y = self.fc(y)
        y = torch.mul(average, y)
        y = y.unsqueeze(2).unsqueeze(2)
        sub = y_abs - y
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        y = torch.mul(torch.sign(x_raw), n_sub)
      
        out = y 
        return out

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)   


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(
                 nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)
        model += [nn.ReflectionPad2d(1),
                  nn.utils.spectral_norm(
                  nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.gmp_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))

        self.model = nn.Sequential(*model)

    def forward(self, input):
        x = self.model(input)

        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))

        heatmap = torch.sum(x, dim=1, keepdim=True)

        x = self.pad(x)
        out = self.conv(x)
        print(out.shape)
        return out
    

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=2, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer)
            if getIntermFeat:                              
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)
        self.fuse_conv = nn.Conv2d(num_D, 1, 1)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def toTenor(self, list):
        num_D = self.num_D
        result = torch.Tensor().to(list[0][0].device)
        for i in range(num_D):
            list[i][0] = torch.nn.functional.interpolate(list[i][0], size=(11, 11), mode='bilinear', align_corners=False)
            result = torch.cat((result, list[i][0]), 1)
        return result
        

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
        return self.toTenor(result)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_nc, 2 * ndf, 1, 1, 0),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2 * ndf, 4 * ndf, 3, 1, 1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(4 * ndf, 1, 3, 1, 1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(2 * ndf, 1, 2, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, input):
        """Standard forward."""
        x1 = self.conv1(input)
        x2 = self.conv2(x1)

        # x3=self.conv2(x2)
        # print (x3.shape)
        x3 = self.conv3(x2)
        x5 = self.net(input)
        # return torch.cat[(x4,x5),1]
     
        return x3 + x5

