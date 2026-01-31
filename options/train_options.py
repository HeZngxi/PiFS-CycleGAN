from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """
    包括train相关的参数，包括：BaseOptions中的参数
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='每训练 save_latest_freq 次迭代，就会保存当前模型的最新状态。这个检查点通常会被命名为 latest_net_G.pth (生成器) 和 latest_net_D.pth (判别器)，方便在训练中断后从最新状态恢复。')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='每训练 save_epoch_freq 个 epoch，就会保存一次模型的检查点。这些检查点通常会带上 epoch 号（例如 5_net_G.pth），用于记录模型在特定 epoch 结束时的状态。')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=100, help='保持初始学习率进行训练的轮数')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='adam优化器的第一个动量参数')
        parser.add_argument('--lr', type=float, default=0.0002, help='adam优化器初始学习率')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='GAN网络的损失函数，lsgan：最小二乘损失；vanilla：原始GAN论文的交叉熵损失；wgangp：Wasserstein GAN with Gradient Penalty 损失')
        parser.add_argument('--pool_size', type=int, default=50, help='图像缓冲池大小')
        parser.add_argument('--lr_policy', type=str, default='linear', help='学习率变化策略 [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='使用step步长衰减时，学习率衰减频率，每间隔lr_decay_iters次迭代进行衰减')

        self.isTrain = True
        return parser
