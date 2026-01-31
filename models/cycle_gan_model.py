import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import os
from PIL import Image


class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    @staticmethod
    # 模型的特定参数，base_options.py中有调用：model_option_setter = models.get_option_setter(model_name)
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument("--lambda_A", type=float, default=10.0, help="weight for cycle loss (A -> B -> A)")
            parser.add_argument("--lambda_B", type=float, default=10.0, help="weight for cycle loss (B -> A -> B)")
            parser.add_argument("--lambda_identity", type=float, default=0.5, help="use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1")
            parser.add_argument("--lambda_label", type=float, default=18, help="weight for label consistency loss")

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.
    
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ["D_A", "G_A", "cycle_A", "idt_A", "D_B", "G_B", "cycle_B", "idt_B", "label_A"]
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ["real_A", "fake_B", "rec_A"]
        visual_names_B = ["real_B", "fake_A", "rec_B"]
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_B(A)
            visual_names_A.append("idt_B")
            visual_names_B.append("idt_A")
    
        if self.isTrain and self.opt.lambda_label > 0.0:
            visual_names_A.append("label_A")
    
        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ["G_A", "G_B", "D_A", "D_B"]
        else:  # during test time, only load Gs
            self.model_names = ["G_A", "G_B"]
    
        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain)
    
        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)
    
        if self.isTrain:
            # 不再需要保存变换函数的引用，因为标签图像已经在数据加载阶段正确处理
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert opt.input_nc == opt.output_nc
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            # self.criterionCycle = torch.nn.L1Loss()
            # self.criterionIdt = torch.nn.L1Loss()

            #自定义混合损失函数，带权重
            # self.criterionCycle = networks.WeightedGlobalL1LocalL2Loss()
            # self.criterionIdt = networks.WeightedGlobalL1LocalL2Loss()
            # 使用原本的L1损失函数
            self.criterionCycle = networks.torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            
            # 使用新的鲁棒有序得分损失函数
            #self.criterionLabel = networks.RobustOrderedScoreLoss(opt.lambda_label)
            # 使用结构一致性损失MIND计算损失
            # self.criterionLabel = networks.MINDLoss()
            # 使用多尺度MIND损失
            self.criterionLabel = networks.MultiMINDLoss(verbose=False, use_gradient_weight=True)  # 添加梯度权重支持

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
    
        Parameters:
            input (dict): include the data itself and its metadata information.
    
        The option 'direction' can be used to swap domain A and B.
        """
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]
        
        # 加载对应的标签图像
        if self.isTrain:
            # 标签图像已经在数据加载阶段进行了与CT图像相同的预处理
            if "A_label" in input:
                self.label_A = input["A_label"].to(self.device)
            else:
                # 如果没有标签图像，则创建一个全零张量
                self.label_A = torch.zeros_like(self.real_A)
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))

    # def backward_D_basic(self, netD, real, fake):
    #     """Calculate GAN loss for the discriminator

    #     Parameters:
    #         netD (network)      -- the discriminator D
    #         real (tensor array) -- real images
    #         fake (tensor array) -- images generated by a generator

    #     Return the discriminator loss.
    #     We also call loss_D.backward() to calculate the gradients.
    #     """
    #     # Real
    #     pred_real = netD(real)
    #     loss_D_real = self.criterionGAN(pred_real, True)
    #     # Fake
    #     pred_fake = netD(fake.detach())
    #     loss_D_fake = self.criterionGAN(pred_fake, False)
    #     # Combined loss and calculate gradients
    #     loss_D = (loss_D_real + loss_D_fake) * 0.5
    #     loss_D.backward()
    #     return loss_D


    def backward_D_basic(self, netD, real, fake):
        """
        计算判别器的GAN损失。
        Calculate GAN loss for the discriminator.

        此版本经过修改，可以同时处理单尺度判别器（返回Tensor）和
        多尺度判别器（返回list of Tensors）。

        参数:
            netD (network)      -- 判别器 D
            real (tensor array) -- 真实图像
            fake (tensor array) -- 生成器生成的图像

        返回判别器损失。
        同时调用 loss_D.backward() 来计算梯度。
        """
        # --- 核心修改部分 ---

        # Real
        pred_real = netD(real)
        # Fake
        pred_fake = netD(fake.detach())

        # 检查判别器的输出是列表（多尺度）还是单个张量（单尺度）
        if isinstance(pred_real, list):
            # --- 多尺度情况 ---
            loss_D_real = 0
            for pred in pred_real:
                loss_D_real += self.criterionGAN(pred, True)

            loss_D_fake = 0
            for pred in pred_fake:
                loss_D_fake += self.criterionGAN(pred, False)

            # 对总损失求平均
            loss_D_real /= len(pred_real)
            loss_D_fake /= len(pred_fake)
        else:
            # --- 单尺度情况 (原始逻辑) ---
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D_fake = self.criterionGAN(pred_fake, False)

        # 组合损失并计算梯度
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    # def backward_G(self):
    #     """Calculate the loss for generators G_A and G_B"""
    #     lambda_idt = self.opt.lambda_identity       #身份判别损失权重默认0.5
    #     lambda_A = self.opt.lambda_A                #循环一致性损失A->B->A权重默认10
    #     lambda_B = self.opt.lambda_B
    #     lambda_label = self.opt.lambda_label        #标签一致性损失权重默认5.0
    #     # Identity loss
    #     if lambda_idt > 0:
    #         # G_A should be identity if real_B is fed: ||G_A(B) - B||
    #         self.idt_A = self.netG_A(self.real_B)
    #         self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
    #         # G_B should be identity if real_A is fed: ||G_B(A) - A||
    #         self.idt_B = self.netG_B(self.real_A)
    #         self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
    #     else:
    #         self.loss_idt_A = 0
    #         self.loss_idt_B = 0

    #     # GAN loss D_A(G_A(A))
    #     self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
    #     # GAN loss D_B(G_B(B))
    #     self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
    #     # Forward cycle loss || G_B(G_A(A)) - A||
    #     self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
    #     # Backward cycle loss || G_A(G_B(B)) - B||
    #     self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        
    #     # Label consistency loss for G_A (CT to EUS generator)
    #     self.loss_label_A = 0
    #     if lambda_label > 0 and hasattr(self, 'label_A') and len(self.label_A) > 0:
    #         # 确保生成的图像和标签图像具有相同的形状
    #         # fake_B是生成的EUS图像，label_A是对应的标签图像
    #         # 通过使用单通道变换，它们应该已经具有相同的形状
    #         if self.fake_B.shape != self.label_A.shape:
    #             print(f"Shape mismatch - fake_B: {self.fake_B.shape}, label_A: {self.label_A.shape}")
            
    #         # 计算生成的EUS图像与标签图像的一致性损失
    #         self.loss_label_A = self.criterionLabel(self.fake_B, self.label_A) * lambda_label

    #     # combined loss and calculate gradients
    #     self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_label_A
    #     self.loss_G.backward()


    # def backward_G(self):
    #     """Calculate the loss for generators G_A and G_B"""
    #     lambda_idt = self.opt.lambda_identity       #身份判别损失权重默认0.5
    #     lambda_A = self.opt.lambda_A                #循环一致性损失A->B->A权重默认10
    #     lambda_B = self.opt.lambda_B
    #     lambda_label = self.opt.lambda_label 
    #     # Identity loss
    #     if lambda_idt > 0:
    #         # G_A should be identity if real_B is fed: ||G_A(B) - B||
    #         self.idt_A = self.netG_A(self.real_B)
    #         self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
    #         # G_B should be identity if real_A is fed: ||G_B(A) - A||
    #         self.idt_B = self.netG_B(self.real_A)
    #         self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
    #     else:
    #         self.loss_idt_A = 0
    #         self.loss_idt_B = 0

    #     # GAN loss D_A(G_A(A))
    #     self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
    #     # GAN loss D_B(G_B(B))
    #     self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
    #     # Forward cycle loss || G_B(G_A(A)) - A||
    #     self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
    #     # Backward cycle loss || G_A(G_B(B)) - B||
    #     self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        
    #     # Label consistency loss for G_A (CT to EUS generator)
    #     self.loss_label_A = 0
    #     if lambda_label > 0 and hasattr(self, 'label_A') and len(self.label_A) > 0:
    #         # 确保生成的图像和标签图像具有相同的形状
    #         # fake_B是生成的EUS图像，label_A是对应的标签图像
    #         # 通过使用单通道变换，它们应该已经具有相同的形状
    #         if self.fake_B.shape != self.label_A.shape:
    #             print(f"Shape mismatch - fake_B: {self.fake_B.shape}, label_A: {self.label_A.shape}")
    #             # 尝试调整形状以匹配
    #             if self.label_A.dim() == 5 and self.fake_B.dim() == 4:
    #                 # 移除多余的维度
    #                 self.label_A = self.label_A.squeeze(1)
    #                 print(f"Adjusted label_A shape: {self.label_A.shape}")
            
    #         # 确保两个张量都是4维的
    #         if self.fake_B.dim() == 4 and self.label_A.dim() == 4:
    #             # 计算生成的EUS图像与标签图像的一致性损失
    #             self.loss_label_A = self.criterionLabel(self.fake_B, self.label_A) * lambda_label
    #         else:
    #             print(f"无法计算标签损失，维度不匹配: fake_B {self.fake_B.dim()}D, label_A {self.label_A.dim()}D")
    #             self.loss_label_A = torch.tensor(0.0, device=self.device)

    #     # combined loss and calculate gradients
    #     self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_label_A
    #     self.loss_G.backward()

    def backward_G(self):
        """
        为生成器 G_A 和 G_B 计算损失。
        Calculate the loss for generators G_A and G_B.

        此版本经过修改，可以正确处理来自多尺度判别器的列表输出，
        并计算平均的对抗性损失。
        """
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_label = self.opt.lambda_label

        # --- 1. 身份损失 (Identity loss) ---
        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # --- 2. 对抗性损失 (GAN loss) --
        # GAN loss D_A(G_A(A))
        pred_fake_B = self.netD_A(self.fake_B)
        # 检查判别器输出是列表（多尺度）还是单个张量（单尺度）
        if isinstance(pred_fake_B, list):
            # 多尺度情况：遍历列表，累加并平均损失
            self.loss_G_A = 0
            for pred in pred_fake_B:
                self.loss_G_A += self.criterionGAN(pred, True)
            self.loss_G_A /= len(pred_fake_B)
        else:
            # 单尺度情况：保持原始逻辑
            self.loss_G_A = self.criterionGAN(pred_fake_B, True)

        # GAN loss D_B(G_B(B))
        pred_fake_A = self.netD_B(self.fake_A)
        if isinstance(pred_fake_A, list):
            # 多尺度情况
            self.loss_G_B = 0
            for pred in pred_fake_A:
                self.loss_G_B += self.criterionGAN(pred, True)
            self.loss_G_B /= len(pred_fake_A)
        else:
            # 单尺度情况
            self.loss_G_B = self.criterionGAN(pred_fake_A, True)

        # --- 3. 循环一致性损失 (Cycle consistency loss) ---
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # --- 4. 标签一致性损失 (Label consistency loss) ---
        self.loss_label_A = 0
        if lambda_label > 0 and hasattr(self, 'label_A') and self.label_A.nelement() > 0:
            if self.fake_B.shape != self.label_A.shape:
                print(f"Shape mismatch - fake_B: {self.fake_B.shape}, label_A: {self.label_A.shape}")
                # 尝试调整形状以匹配
                if self.label_A.dim() == 5 and self.fake_B.dim() == 4:
                    self.label_A = self.label_A.squeeze(1)
                    print(f"Adjusted label_A shape: {self.label_A.shape}")

            if self.fake_B.dim() == 4 and self.label_A.dim() == 4:
                # 将标签图像作为额外参数传递给MultiMIND损失函数
                self.loss_label_A = self.criterionLabel(self.fake_B, self.label_A, self.label_A) * lambda_label
            else:
                print(f"无法计算标签损失，维度不匹配: fake_B {self.fake_B.dim()}D, label_A {self.label_A.dim()}D")
                self.loss_label_A = torch.tensor(0.0, device=self.device)

        # --- 5. 组合总损失并反向传播 ---
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_label_A
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights