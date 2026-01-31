import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np

class Identity(nn.Module):
    def forward(self, x):
        return x


# 归一化层选择
def get_norm_layer(norm_type="instance"):
    """
    根据参数选择，返回不同的归一化层配置。

    参数:
        norm_type (str) -- 归一化层类型: batch | syncbatch | instance | none

    返回:
        norm_layer (function) -- 归一化层的构造函数
    """
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == "syncbatch":
        norm_layer = functools.partial(nn.SyncBatchNorm, affine=True, track_running_stats=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == "none":

        def norm_layer(x):
            return Identity()

    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == "linear":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError("learning rate policy [%s] is not implemented", opt.lr_policy)
    return scheduler


def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

# 网络初始化
def init_net(net, init_type="normal", init_gain=0.02):
    """Initialize a network: 1. register CPU/GPU device; 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.

    Return an initialized network.
    """
    import os

    if torch.cuda.is_available():
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            net.to(local_rank)
            print(f"Initialized with device cuda:{local_rank}")
        else:
            net.to(0)
            print("Initialized with device cuda:0")
    init_weights(net, init_type, init_gain=init_gain)
    return net



import re # 导入正则表达式模块，用于解析n_blocks
def define_G(input_nc, output_nc, ngf, netG, norm="batch", use_dropout=False, init_type="normal", init_gain=0.02):
    """
    创建一个生成器。

    此函数现在优先支持并默认推荐使用 'resnet' 架构，以及新增的 'resnet_fem' 架构。

    参数:
        input_nc (int): 输入图像的通道数。
        output_nc (int): 输出图像的通道数。
        ngf (int): 网络第一层卷积的基础通道数。
        netG (str): 生成器架构的名称。
                    - 经典: 'resnet_9blocks', 'resnet_6blocks'
                    - D2Net: 'd2net_resnet_3blocks' (或 6blocks, 9blocks)
                    - 增强ResNet: 'resnet_fem_9blocks' (或其他blocks数量)
                    - U-Net: 'unet_128', 'unet_256'
        norm (str): 归一化层的类型: 'batch' | 'instance' | 'none'。
        use_dropout (bool): 是否在ResnetBlock中使用Dropout。
        init_type (str): 初始化方法: 'normal' | 'xavier' | 'kaiming' | 'orthogonal'。
        init_gain (float): 初始化方法的缩放因子。

    返回:
        一个初始化好的生成器网络。
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    
    if 'resnet' in netG and 'fem' not in netG and 'd2net' not in netG:  # 经典ResNet
        match = re.search(r'(\d+)blocks', netG)
        if match:
            n_blocks = int(match.group(1))
        else:
            n_blocks = 9 # 原始默认值
            print(f"Warning: n_blocks not specified in netG='{netG}'. Defaulting to {n_blocks} blocks.")

        print(f"Using classic ResnetGenerator with {n_blocks} ResnetBlocks.")
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks)

    elif 'd2net_resnet' in netG:  # D2Net ResNet
        match = re.search(r'(\d+)blocks', netG)
        if match:
            n_blocks = int(match.group(1))
        else:
            n_blocks = 3 
            print(f"Warning: n_blocks not specified in netG='{netG}'. Defaulting to {n_blocks} blocks.")
            
        print(f"Using D2NetResnetGenerator with {n_blocks} ResnetBlocks.")
        net = D2NetResnetGenerator(input_nc, output_nc, ngf, n_blocks=n_blocks, norm_layer=norm_layer, use_dropout=use_dropout)

    elif 'resnet_fem' in netG:  # 带FEM的ResNet
        match = re.search(r'(\d+)blocks', netG)
        if match:
            n_blocks = int(match.group(1))
        else:
            n_blocks = 9 
            print(f"Warning: n_blocks not specified in netG='{netG}'. Defaulting to {n_blocks} blocks.")
            
        print(f"Using ResnetGeneratorWithFEM with {n_blocks} ResnetBlocks.")
        net = ResnetGeneratorWithFEM(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks)

    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    
    return init_net(net, init_type, init_gain=init_gain)




class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode        # 指定损失函数选择
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ["wgangp"]:
            self.loss = None
        else:
            raise NotImplementedError("gan mode %s not implemented" % gan_mode)

    # 根据图像真假设置标签 1/0
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

    # 计算损失值。根据判别器的输出预测和图像的真假标签，按照指定的GAN模式（lsgan/vanilla/wgangp）计算对应的损失值。
    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == "wgangp":
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


    # 计算WGAN-GP中的梯度惩罚损失，用于确保判别器满足Lipschitz连续性条件，使GAN训练更加稳定
    # 在WGAN-GP模式下，损失等于GANLoss函数中__call__函数计算的基本损失+这里的梯度惩罚损失

def cal_gradient_penalty(netD, real_data, fake_data, device, type="mixed", constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == "real":  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == "fake":
            interpolatesv = fake_data
        elif type == "mixed":
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError(f"{type} not implemented")
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv, grad_outputs=torch.ones(disc_interpolates.size()).to(device), create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """
    基于ResNet（残差网络）架构设计的生成器类
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type="reflect"):
        """
        Parameters:
            input_nc (int)      -- 输入图像的通道数
            output_nc (int)     -- 输出图像的通道数
            ngf (int)           -- 最后一个卷积层的滤波器数量
            norm_layer          -- 归一化层
            use_dropout (bool)  -- 是否使用 dropout 层
            n_blocks (int)      -- ResNet 块的数量
            padding_type (str)  -- 卷积层中使用的填充类型: reflect | replicate | zero
        """
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(ngf * mult * 2), nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
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
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in the input image
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)



class MultiScaleDiscriminator(nn.Module):
    """
    一个多尺度的联合判别器，包含三个在不同图像尺度上操作的PatchGAN。
    - D1: 在原始图像上操作，关注高频细节。
    - D2: 在1/2下采样的图像上操作，关注中层结构。
    - D3: 在1/4下采样的图像上操作，关注全局一致性。
    """
    def __init__(self, input_nc, ndf=64, n_layers=4, norm_layer=nn.BatchNorm2d):
        """
        初始化多尺度判别器。
        参数:
            input_nc (int): 输入图像的通道数。
            ndf (int): 第一个卷积层的基础通道数。
            n_layers (int): 每个PatchGAN判别器的层数。根据您的要求，默认为4。
            norm_layer (nn.Module): 归一化层的类型。
        """
        super(MultiScaleDiscriminator, self).__init__()
        
        # --- 定义三个PatchGAN判别器 ---
        # 它们的结构完全相同，但权重是独立训练的
        self.discriminator_1x = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer)
        self.discriminator_2x = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer)
        self.discriminator_4x = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer)
        
        # --- 定义下采样层 ---
        # 使用平均池化(Average Pooling)进行下采样。这是一种简单且常用的方法，
        #因为它不会引入额外的可学习参数。
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, x):
        """
        定义前向传播逻辑。
        
        参数:
            x (Tensor): 原始分辨率的输入图像。
        
        返回:
            list[Tensor]: 一个包含三个判别器输出结果的列表。
                         训练循环需要处理这个列表来计算总损失。
        """
        results = []
        
        # 判别器1: 在原始尺度上进行判别
        results.append(self.discriminator_1x(x))
        
        # 下采样到1/2尺度
        x_2x = self.downsample(x)
        # 判别器2: 在1/2尺度上进行判别
        results.append(self.discriminator_2x(x_2x))
        
        # 下采样到1/4尺度
        x_4x = self.downsample(x_2x)
        # 判别器3: 在1/4尺度上进行判别
        results.append(self.discriminator_4x(x_4x))
        
        return results





class FGFE(nn.Module):
    """
    Fourier-based Global Feature Extraction (FGFE) Module.
    基于傅里叶的全局特征提取模块。

    该模块严格按照论文图3(a)的设计实现，旨在高效地捕捉特征的全局依赖关系。
    它通过在频域中进行特征交互，避免了传统自注意力机制在高分辨率图像上
    带来的巨大内存和计算开销。

    模块流程:
    1. 输入特征图 x 并行地送入三个独立的分支，分别生成 Q (Query), K (Key), V (Value)。
    2. 每个分支内部由一个 1x1 卷积（用于通道融合）和一个 3x3 深度卷积（用于空间特征提取）组成。
    3. Q 和 K 经过傅里叶变換（FFT）被映射到频域。
    4. 在频域中，Q 和 K 的频域表示进行元素级乘法，以计算全局相关性。
    5. 将计算结果通过傅里叶逆变换（iFFT）映射回空间域，得到空间注意力图。
    6. 该空间注意力图与 V 进行元素级乘法，实现对 V 特征的全局上下文调制。
    7. 最后，通过一个 1x1 卷积进行最终的特征融合与提炼，输出结果。
    """
    def __init__(self, channels):
        """
        初始化FGFE模块的各个层。

        参数:
            channels (int): 输入和输出的特征通道数。模块内部保持通道数不变。
        """
        super(FGFE, self).__init__()

        # --- 1. 定义用于生成 Q, K, V 的卷积块 ---
        # 根据图示和分析，每个分支都包含一个 "1x1 Conv" 和一个 "3x3 DC" (Depthwise Conv)。
        # 这种 "Pointwise -> Depthwise" 的组合是高效的特征提取方式。
        # Q, K, V 三个分支的权重是独立的，不共享。

        # 定义 Q (Query) 的生成网络
        self.query_conv = nn.Sequential(
            # 第一个 1x1 卷积：作为 Pointwise Convolution，负责通道信息的线性投影和融合。
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            # 第二个 3x3 深度卷积 (Depthwise Convolution)：
            # `groups=channels` 是实现深度卷积的关键，它让每个输入通道都拥有自己独立的卷积核，
            # 专注于提取空间特征，而不混合通道信息。`padding=1` 保持空间尺寸不变。
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, groups=channels)
        )

        # 定义 K (Key) 的生成网络，结构与 Q 相同，但权重独立。
        self.key_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, groups=channels)
        )

        # 定义 V (Value) 的生成网络，结构与 Q, K 相同，但权重独立。
        self.value_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, groups=channels)
        )

        # --- 2. 定义最终的输出卷积层 ---
        # 这个 1x1 卷积在注意力加权之后，用于对特征进行最后的融合，类似于 Transformer 中的输出投影层。
        self.output_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)


    def forward(self, x):
        """
        定义FGFE模块的前向传播逻辑。

        参数:
            x (Tensor): 输入的特征图，形状为 (B, C, H, W)。
                        B=Batch size, C=Channels, H=Height, W=Width.

        返回:
            Tensor: 经过全局特征提取后输出的特征图，形状与输入相同 (B, C, H, W)。
        """
        # 步骤 1: 将输入 x 通过各自的卷积块，生成 Q, K, V
        # 输出的 Q, K, V 形状都与输入 x 相同: (B, C, H, W)
        q = self.query_conv(x)
        k = self.key_conv(x)
        v = self.value_conv(x)

        # 步骤 2: 将 Q 和 K 变换到频域
        # 使用 torch.fft.fft2 对最后两个维度 (H, W) 进行2D快速傅里叶变换。
        q_fft = torch.fft.fft2(q, dim=(-2, -1))
        k_fft = torch.fft.fft2(k, dim=(-2, -1))

        # 步骤 3: 在频域中计算全局相关性
        # 通过元素级乘法实现。这在计算上远比空间域的点积（矩阵乘法）高效。
        # 使用 K 的共轭复数 (torch.conj) 是傅里叶变换中计算相关性的标准做法，
        # 它等效于空间域中的卷积/相关操作。
        attention_fft = q_fft * torch.conj(k_fft)

        # 步骤 4: 将计算出的相关性从频域逆变换回空间域
        # 使用 torch.fft.ifft2 进行2D快速傅里叶逆变换。
        # 逆变换的结果是一个复数张量，我们取其实部 (.real) 作为最终的空间注意力图。
        spatial_attention = torch.fft.ifft2(attention_fft, dim=(-2, -1)).real

        # 步骤 5: 将空间注意力图应用到 V 上
        # 通过元素级乘法，将学习到的全局上下文信息（注意力权重）施加到 V 的特征上。
        # 这会增强重要特征并抑制不相关特征。
        attended_v = spatial_attention * v

        # 步骤 6: 通过最后的 1x1 卷积进行特征融合
        # 对加权后的特征进行最后一次提炼，并输出最终结果。
        output = self.output_conv(attended_v)

        return output
    

class MLFE(nn.Module):
    """
    Multi-scale Local Feature Extraction (MLFE) Module.
    多尺度局部特征提取模块。

    该模块严格按照论文图3(b)的设计实现。其核心思想是通过并行处理多个分支，
    每个分支使用不同形状的深度卷积核（Depthwise Convolution, DC）来捕捉
    不同尺度和方向的局部特征。这种设计对于处理细节丰富的高分辨率图像
    非常有效。

    模块流程:
    1. Split: 将输入的特征图沿通道维度均分为四个部分。
    2. Parallel Branches: 四个部分被分别送入四个平行的分支进行处理：
        - Branch 1: 3x3 深度卷积，用于捕捉常规的方形局部模式。
        - Branch 2: 11x1 深度卷积，用于捕捉垂直方向的条形特征。
        - Branch 3: 1x11 深度卷积，用于捕捉水平方向的条形特征。
        - Branch 4: Identity，恒等映射，直接保留原始特征。
    3. Concat: 将四个分支的输出结果沿通道维度重新拼接起来，恢复原始通道数。
    """
    def __init__(self, channels):
        """
        初始化MLFE模块的各个分支。

        参数:
            channels (int): 输入特征图的总通道数。
                          注意：此模块要求输入通道数必须是4的倍数。
        """
        super(MLFE, self).__init__()

        # --- 1. 验证输入通道数 ---
        # 由于输入将被均分为4个分支，因此总通道数必须能被4整除。
        assert channels % 4 == 0, "Input channels for MLFE module must be divisible by 4."

        # 计算每个分支处理的通道数
        self.branch_channels = channels // 4

        # --- 2. 定义三个卷积分支 ---
        # 所有分支都使用深度卷积（DC），通过设置 `groups = branch_channels` 来实现。
        # 深度卷积为每个输入通道分配一个独立的卷积核，只提取空间特征，不混合通道信息，计算效率极高。

        # Branch 1: 3x3 深度卷积
        self.branch_3x3 = nn.Conv2d(
            in_channels=self.branch_channels,
            out_channels=self.branch_channels,
            kernel_size=3,
            padding=1,  # padding = (kernel_size - 1) / 2 = (3-1)/2 = 1
            groups=self.branch_channels
        )

        # Branch 2: 11x1 深度卷积 (垂直条形核)
        self.branch_11x1 = nn.Conv2d(
            in_channels=self.branch_channels,
            out_channels=self.branch_channels,
            kernel_size=(11, 1),
            padding=(5, 0), # H方向padding=(11-1)/2=5, W方向padding=(1-1)/2=0
            groups=self.branch_channels
        )

        # Branch 3: 1x11 深度卷积 (水平条形核)
        self.branch_1x11 = nn.Conv2d(
            in_channels=self.branch_channels,
            out_channels=self.branch_channels,
            kernel_size=(1, 11),
            padding=(0, 5), # H方向padding=(1-1)/2=0, W方向padding=(11-1)/2=5
            groups=self.branch_channels
        )
        
        # Branch 4 (Identity) 是一个直通连接，不需要在 __init__ 中定义任何层。

    def forward(self, x):
        """
        定义MLFE模块的前向传播逻辑。

        参数:
            x (Tensor): 输入的特征图，形状为 (B, C, H, W)。

        返回:
            Tensor: 经过多尺度局部特征提取后输出的特征图，形状与输入相同 (B, C, H, W)。
        """
        # 步骤 1: Split
        # 使用 torch.chunk 将输入张量 x 沿通道维度 (dim=1) 分成4个块。
        # 返回的是一个包含4个张量的元组，每个张量的形状为 (B, C/4, H, W)。
        chunks = x.chunk(4, dim=1)
        
        # 将元组解包到各个分支的输入
        x_b1, x_b2, x_b3, x_b4 = chunks

        # 步骤 2: Parallel Branches Processing
        # 每个块分别通过对应的分支
        out_b1 = self.branch_3x3(x_b1)
        out_b2 = self.branch_11x1(x_b2)
        out_b3 = self.branch_1x11(x_b3)
        # 第四个分支是恒等映射，所以其输出就是其输入
        out_b4 = x_b4

        # 步骤 3: Concat
        # 使用 torch.cat 将四个分支的输出张量沿通道维度 (dim=1) 重新拼接起来。
        # 输出的形状恢复为 (B, C, H, W)。
        output = torch.cat([out_b1, out_b2, out_b3, out_b4], dim=1)

        return output
    

class FFN(nn.Module):
    """
    Feedforward Network (FFN) Module.
    前馈网络模块。

    该模块严格按照论文图3(c)和第3.6节的描述实现。它是一个标准的特征变换单元，
    通过一个扩张-收缩结构来对特征进行非线性映射和提炼。

    模块流程:
    1. 扩张层: 一个 3x3 卷积层，用于将输入的通道维度扩大数倍（由 expansion_factor 控制），
               同时进行空间特征的提取。
    2. 激活层: 使用 GELU (Gaussian Error Linear Unit) 激活函数引入非线性。
    3. 收缩层: 一个 1x1 卷积层，将通道维度压缩回原始大小，完成特征的融合与投影。
    """
    def __init__(self, channels, expansion_factor=2.0):
        """
        初始化FFN模块。

        参数:
            channels (int): 输入和输出的特征通道数。
            expansion_factor (float, optional): 
                通道扩张因子，决定了中间隐藏层的通道数。默认为 2.0，
                这是Transformer等架构中的常见设置。
        """
        super(FFN, self).__init__()
        hidden_channels = int(channels * expansion_factor)

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=channels, 
                out_channels=hidden_channels, 
                kernel_size=3, 
                padding=1
            ),

            # 非线性激活层 (Activation Layer)
            nn.GELU(),

            # 收缩层 (Contraction Layer)
            nn.Conv2d(
                in_channels=hidden_channels, 
                out_channels=channels, 
                kernel_size=1
            )
        )

    def forward(self, x):
        """
        定义FFN模块的前向传播逻辑。

        参数:
            x (Tensor): 输入的特征图，形状为 (B, C, H, W)。

        返回:
            Tensor: 经过非线性变换后输出的特征图，形状与输入相同 (B, C, H, W)。
        """
        return self.net(x)
    

class AFMM(nn.Module):
    """
    Adaptive Feature Modulation Module (AFMM).
    自适应特征调制模块。

    该模块严格按照论文中详细的AFMM结构图实现，用于智能地融合U-Net架构中的
    编码器特征（来自skip connection）和解码器特征（来自上采样）。
    它通过学习一个共享的注意力图来同时调制两个输入，然后将它们相加。

    模块流程:
    1. 两个输入（encoder_features, decoder_features）首先各自通过一个独立的1x1卷积进行特征投影。
    2. 将两个投影后的特征拼接（Concat）起来。
    3. 将拼接后的特征通过另一个1x1卷积来生成一个原始的注意力图。
    4. 将原始注意力图通过Softmax激活函数，得到最终的归一化注意力权重。
    5. 使用这个共享的注意力权重，通过元素级乘法，分别调制两个投影后的特征。
    6. 最后，将两个被调制过的特征相加，得到最终的融合结果。
    """
    def __init__(self, channels):
        """
        初始化AFMM模块。

        参数:
            channels (int): 输入特征图的通道数。模块假定两个输入的通道数相同。
        """
        super(AFMM, self).__init__()

        # 编码器特征的投影层
        self.conv_encoder = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        # 解码器特征的投影层
        self.conv_decoder = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.attention_generator = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=1)


    def forward(self, encoder_features, decoder_features):
        """
        定义AFMM模块的前向传播逻辑。

        参数:
            encoder_features (Tensor): 来自编码器的特征图 (skip connection)。
                                       形状为 (B, C, H, W)。
            decoder_features (T ensor): 来自解码器的特征图 (upsampled)。
                                       形状为 (B, C, H, W)。
        
        返回:
            Tensor: 融合后的特征图，形状为 (B, C, H, W)。
        """
        # 步骤 1: 对两个输入进行独立的1x1卷积投影
        processed_encoder = self.conv_encoder(encoder_features)
        processed_decoder = self.conv_decoder(decoder_features)

        # 步骤 2: 拼接投影后的特征
        # 沿通道维度(dim=1)拼接，形状变为 (B, C*2, H, W)
        combined_features = torch.cat([processed_encoder, processed_decoder], dim=1)

        # 步骤 3: 生成并激活注意力图
        # a. 通过1x1卷积生成原始注意力图，形状为 (B, C, H, W)
        attention_raw = self.attention_generator(combined_features)
        # b. 通过Softmax激活，得到最终的注意力权重图
        attention_map = self.softmax(attention_raw)

        # 步骤 4: 特征调制
        # 使用同一个注意力图，分别调制两个投影后的特征
        # alpha (α)
        modulated_encoder = processed_encoder * attention_map
        # beta (β)
        modulated_decoder = processed_decoder * attention_map

        # 步骤 5: 最终融合
        # 将两个调制后的特征相加
        output = modulated_encoder + modulated_decoder

        return output
    

class FeatureExtractionModule(nn.Module):
    """
    特征提取模块 (Feature Extraction Module, FEM)。

    该模块严格遵循论文中右下角的详细放大图进行实现。它由两个串联的残差块构成，
    分别是一个基于注意力的块（FGFE + MLFE）和一个前馈网络块（FFN）。
    图中的LN（Layer Normalization）根据要求被替换为Instance Normalization。

    模块流程:
    1. 第一个残差块 (注意力块):
       - 输入 -> InstanceNorm -> 并行(FGFE, MLFE) -> Concat -> 1x1 Conv -> + (与原始输入相加)
    2. 第二个残差块 (前馈网络块):
       - 上一阶段的输出 -> InstanceNorm -> FFN -> + (与上一阶段的输出相加)
    """
    def __init__(self, channels):
        """
        初始化FeatureExtractionModule。

        参数:
            channels (int): 输入、输出以及模块内部主要路径的特征通道数。
        """
        super(FeatureExtractionModule, self).__init__()

        self.norm1 = nn.InstanceNorm2d(channels, affine=True)

        self.fgfe = FGFE(channels)
        self.mlfe = MLFE(channels)

        self.fusion_conv = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1)

        self.norm2 = nn.InstanceNorm2d(channels, affine=True)
        
        self.ffn = FFN(channels)


    def forward(self, x):
        """
        定义FEM模块的前向传播逻辑。

        参数:
            x (Tensor): 输入的特征图，形状为 (B, C, H, W)。
        
        返回:
            Tensor: 经过特征提取后输出的特征图，形状与输入相同 (B, C, H, W)。
        """
        
        # 保存原始输入，用于第一个残差连接
        residual1 = x

        # a. 归一化
        x_norm1 = self.norm1(x)
        
        # b. 并行通过 FGFE 和 MLFE
        x_fgfe = self.fgfe(x_norm1)
        x_mlfe = self.mlfe(x_norm1)

        # c. 拼接并融合
        x_cat = torch.cat([x_fgfe, x_mlfe], dim=1) # 形状变为 (B, 2*C, H, W)
        x_fused = self.fusion_conv(x_cat)         # 形状恢复为 (B, C, H, W)

        # d. 添加第一个残差连接
        x_res1 = residual1 + x_fused

        # --- 第二个残差块 ---

        # 保存第一个块的输出，用于第二个残差连接
        residual2 = x_res1

        # a. 归一化
        x_norm2 = self.norm2(x_res1)
        
        # b. 通过 FFN
        x_ffn = self.ffn(x_norm2)

        # c. 添加第二个残差连接
        output = residual2 + x_ffn

        return output
    

# 两层下采样D2Net
class D2NetResnetGenerator(nn.Module):
    """
    一个融合了D2Net模块和经典ResNet结构的生成器。
    它采用了一个对称的【两层】下采样/上采样U-Net架构。

    - 编码器: 使用 FeatureExtractionModule (FEM) 进行特征增强，后接一个步进卷积进行下采样。
    - 瓶颈层: 由多个经典的 ResnetBlock 串联而成，用于深度特征变换。
    - 解码器: 使用转置卷积进行上采样，然后通过 AFMM 模块智能融合来自编码器的skip-connection特征。
    """
    def __init__(self, input_nc=1, output_nc=1, ngf=64, n_blocks=9, norm_layer=nn.InstanceNorm2d, use_dropout=False, padding_type='reflect'):
        """
        初始化生成器。
        参数:
            input_nc (int): 输入图像的通道数。
            output_nc (int): 输出图像的通道数。
            ngf (int): 网络第一层卷积后的基础通道数。
            n_blocks (int): 瓶颈层中ResnetBlock的数量。
            norm_layer (nn.Module): 归一化层的类型。
            use_dropout (bool): 是否在ResnetBlock中使用Dropout。
        """
        super(D2NetResnetGenerator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # 输入处理层
        self.initial_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        )
        
        # --- 编码器: 2层下采样 ---
        self.encoder_fem1 = FeatureExtractionModule(ngf)
        self.down1 = nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias)
        self.encoder_fem2 = FeatureExtractionModule(ngf * 2)
        self.down2 = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=use_bias)

        # --- 瓶颈层 ---
        # 输入通道数调整为 ngf * 4，以匹配第二层下采样的输出
        bottleneck = []
        for i in range(n_blocks):
            bottleneck += [ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.bottleneck = nn.Sequential(*bottleneck)

        # --- 解码器: 2层上采样 ---
        # 输入通道数调整为 ngf * 4，以匹配瓶颈层的输出
        self.up1 = nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)
        self.decoder_afmm1 = AFMM(ngf * 2)
        self.up2 = nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)
        self.decoder_afmm2 = AFMM(ngf)
        
        # 输出处理层
        self.output_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        # 编码器
        x0 = self.initial_conv(x)
        e1 = self.encoder_fem1(x0)
        d1 = self.down1(e1)
        e2 = self.encoder_fem2(d1)
        d2 = self.down2(e2)
        
        # 瓶颈层
        b = self.bottleneck(d2)
        
        # 解码器
        u1 = self.up1(b)
        f1 = self.decoder_afmm1(encoder_features=e2, decoder_features=u1)
        u2 = self.up2(f1)
        f2 = self.decoder_afmm2(encoder_features=e1, decoder_features=u2)
        
        # 输出
        return self.output_conv(f2)
    
# 一层FEM
class ResnetGeneratorWithFEM(nn.Module):
    """
    基于ResNet（残差网络）架构设计的生成器类，
    在初始特征提取后加入一个FeatureExtractionModule。
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=None, use_dropout=False, n_blocks=9, padding_type="reflect"):
        """
        Parameters:
            input_nc (int)      -- 输入图像的通道数
            output_nc (int)     -- 输出图像的通道数
            ngf (int)           -- 最后一个卷积层的滤波器数量
            norm_layer          -- 归一化层 (应为functools.partial或nn.InstanceNorm2d/nn.BatchNorm2d)
            use_dropout (bool)  -- 是否使用 dropout 层
            n_blocks (int)      -- ResNet 块的数量
            padding_type (str)  -- 卷积层中使用的填充类型: reflect | replicate | zero
        """
        assert n_blocks >= 0
        super(ResnetGeneratorWithFEM, self).__init__()
        
        # 判断是否需要偏置，与get_norm_layer的逻辑一致
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d # 假设 norm_layer 是 nn.InstanceNorm2d 或 nn.BatchNorm2d

        self.initial_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        )
        
        # --- 在这里插入FeatureExtractionModule ---
        # FEM的输入和输出通道数都与ngf相同
        # 注意：这里直接调用了您脚本中已有的 FeatureExtractionModule
        self.fem = FeatureExtractionModule(ngf) 
        # ----------------------------------------

        n_downsampling = 2 # 两次下采样
        # 第一次下采样
        self.down1 = nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias)
        # 第二次下采样
        self.down2 = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=use_bias)

        mult = 2**n_downsampling
        bottleneck = []
        for i in range(n_blocks):  # 添加ResNet块 (根据n_blocks参数决定数量)
            # 注意：这里直接调用了您脚本中已有的 ResnetBlock
            bottleneck += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.bottleneck = nn.Sequential(*bottleneck)

        # 上采样和AFMM模块
        self.up1 = nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)
        self.afmm1 = AFMM(ngf * 2)  # AFMM模块融合编码器特征
        self.up2 = nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)
        self.afmm2 = AFMM(ngf)  # AFMM模块融合编码器特征

        self.output_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, input):
        """Modified forward to include skip connections with AFMM"""
        # 初始卷积
        x0 = self.initial_conv(input)
        # 经过FEM模块
        x_fem = self.fem(x0)
        
        # 下采样
        d1 = self.down1(x_fem)
        d2 = self.down2(d1)
        
        # 瓶颈层
        b = self.bottleneck(d2)
        
        # 上采样与AFMM融合
        u1 = self.up1(b)
        f1 = self.afmm1(encoder_features=d1, decoder_features=u1)  # 融合第一次下采样的特征
        u2 = self.up2(f1)
        f2 = self.afmm2(encoder_features=x_fem, decoder_features=u2)  # 融合FEM后的特征
        
        # 输出
        return self.output_conv(f2)

