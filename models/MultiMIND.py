import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiMIND(nn.Module):
    """
    用于计算多尺度MIND值的类
    """
    
    def __init__(self, patch_size=7, non_local_size=9, gaussian_sigma=2.0, use_gradient_weight=False):
        """
        初始化MIND计算模块
        
        参数:
        - patch_size: 用于聚合的patch大小
        - non_local_size: 非局部区域大小
        - gaussian_sigma: 高斯核的标准差
        - use_gradient_weight: 是否使用梯度权重图进行加权
        """
        super(MultiMIND, self).__init__()
        self.patch_size = patch_size
        self.non_local_size = non_local_size
        self.gaussian_sigma = gaussian_sigma
        self.use_gradient_weight = use_gradient_weight
        
        # 预计算高斯核
        self.kernel = self._create_gaussian_kernel()
    
    def _create_gaussian_kernel(self):
        """
        创建高斯核
        """
        size = self.patch_size
        sigma = self.gaussian_sigma
        ax = torch.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size, size)
    
    def _calculate_mind_single(self, image):
        """
        计算单张图像的MIND特征
        
        参数:
        - image: 输入图像，形状为 (B, C, H, W)
        """
        if image.shape[1] > 1:
            image = torch.mean(image, dim=1, keepdim=True)
        
        # 将图像从[-1, 1]范围转换到[0, 255]范围
        image = (image * 0.5 + 0.5) * 255.0
        
        batch_size, channels, height, width = image.shape
        device = image.device

        kernel = self.kernel.to(device)
        
        # 计算特征维度
        feature_dim = self.non_local_size * self.non_local_size - 1
        mind_features = []

        search_radius = self.non_local_size // 2
        
        # 填充图像以便处理边界
        pad_len = search_radius
        image_padded = F.pad(image, (pad_len, pad_len, pad_len, pad_len), mode='reflect')
        
        # 存储所有的 Dp(I, x, x+alpha)
        dp_maps = {} 
        
        # 计算所有位移的 patch 距离
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                if dy == 0 and dx == 0:
                    continue
                img_center = image 
                
                # 获取偏移后的图像
                y_start = pad_len + dy
                x_start = pad_len + dx
                img_shifted = image_padded[:, :, y_start : y_start+height, x_start : x_start+width]
                
                # 计算像素差的平方
                diff_sq = (img_center - img_shifted) ** 2
                
                # 使用高斯核进行卷积聚合
                dp = F.conv2d(diff_sq, kernel, padding=self.patch_size//2)
                
                dp_maps[(dy, dx)] = dp
                mind_features.append(dp)
        
        # 将列表堆叠为张量 (B, Feature_Dim, H, W)
        mind_features = torch.cat(mind_features, dim=1)
        
        # 计算局部方差 Vx
        vx_sum = torch.zeros_like(image)
        count = 0
        
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                if (dy, dx) in dp_maps:
                    vx_sum += dp_maps[(dy, dx)]
                    count += 1
        
        # 计算平均值，加一个小 epsilon 防止除零
        vx = vx_sum / max(count, 1)
        vx = vx + 1e-6
        
        mind_final = torch.exp(-mind_features / vx)
        
        return mind_final
    
    def forward(self, label_image, eus_image, weight_map=None):
        """
        计算两幅图像之间的MIND损失
        
        参数:
        - label_image: 标签图像 (物理模拟的EUS图像)，形状为 (B, C, H, W)
        - eus_image: 真实EUS图像，形状为 (B, C, H, W)
        - weight_map: 预计算的梯度权重图，形状为 (B, C, H, W)
        """
        # 计算两幅图像的MIND特征
        mind_label = self._calculate_mind_single(label_image)
        mind_eus = self._calculate_mind_single(eus_image)
        
        # 计算MIND特征之间的差异
        mind_diff = torch.abs(mind_label - mind_eus)
        
        # 如果需要使用梯度权重
        if self.use_gradient_weight and weight_map is not None:
            # 使用权重图对MIND差异进行加权
            # 扩展权重图到MIND特征的维度
            weight_map_expanded = weight_map.expand_as(mind_diff)
            weighted_mind_diff = mind_diff * weight_map_expanded
            
            # 对特征维度和空间维度求平均，得到每个样本的损失值
            loss_per_sample = torch.mean(weighted_mind_diff, dim=(1, 2, 3))
        else:
            # 使用全局平均
            loss_per_sample = torch.mean(mind_diff, dim=(1, 2, 3))
        
        # 返回批次平均损失
        loss = torch.mean(loss_per_sample)
        
        return loss, mind_label, mind_eus
    

    def get_diff_map(self, label_image, eus_image, weight_map=None):
        # 计算两幅图像的MIND特征
        mind_label = self._calculate_mind_single(label_image)
        mind_eus = self._calculate_mind_single(eus_image)
        
        # 计算MIND特征之间的差异 (B, Feature_Dim, H, W)
        mind_diff = torch.abs(mind_label - mind_eus)
        
        if self.use_gradient_weight and weight_map is not None:
            # 使用权重图对MIND差异进行加权
            weight_map_expanded = weight_map.expand_as(mind_diff)
            mind_diff = mind_diff * weight_map_expanded
        
        # 对特征维度求平均，得到空间差异图 (B, 1, H, W)
        diff_map = torch.mean(mind_diff, dim=1, keepdim=True)
        
        return diff_map


def compute_multiscale_mind(label_image, eus_image, scales, weights, patch_sizes, non_local_sizes, gaussian_sigma=2.0, verbose=False, use_gradient_weight=False, label_for_weight=None):
    """
    计算多尺度MIND损失
    
    参数:
    - label_image: 标签图像 (B, C, H, W)
    - eus_image: EUS图像 (B, C, H, W)
    - scales: 尺度列表，例如 [1, 0.25, 0.0625, 0.015625] 表示原始、1/4、1/16、1/64
    - weights: 每个尺度的权重，例如 [0.5, 0.3, 0.2]
    - patch_sizes: 每个尺度对应的patch_size
    - non_local_sizes: 每个尺度对应的non_local_size
    - gaussian_sigma: 高斯核的标准差
    - verbose: 是否输出调试信息
    - use_gradient_weight: 是否使用梯度权重
    - label_for_weight: 用于计算梯度权重的标签图像
    """
    total_loss = 0.0
    device = label_image.device
    
    # 预先计算梯度权重图，只在原始分辨率上计算一次
    weight_map = None
    if use_gradient_weight and label_for_weight is not None:
        from .GradientWeightMap import GradientWeightMap
        weight_map_generator = GradientWeightMap().to(label_for_weight.device)
        # 将标签图像从[-1,1]范围转换到[0,1]范围，以便计算梯度权重
        normalized_label = (label_for_weight + 1.0) / 2.0
        weight_map = weight_map_generator(normalized_label)
        
        # 使用高斯模糊处理权重图（模拟GS_preprocessLabel的功能）
        weight_map = _gaussian_blur_global(weight_map)
    
    if verbose:
        print(f"MultiMIND Loss breakdown:")
    
    for i, (scale, weight, patch_size, non_local_size) in enumerate(zip(scales, weights, patch_sizes, non_local_sizes)):
        if scale == 1.0:  # 原始尺寸
            scaled_label = label_image
            scaled_eus = eus_image
            scaled_weight_map = weight_map
        else:
            # 计算新的尺寸
            new_height = max(int(label_image.shape[2] * scale), 1)
            new_width = max(int(label_image.shape[3] * scale), 1)
            
            # 使用双线性插值进行缩放
            scaled_label = F.interpolate(label_image, size=(new_height, new_width), mode='bilinear', align_corners=False)
            scaled_eus = F.interpolate(eus_image, size=(new_height, new_width), mode='bilinear', align_corners=False)
            
            # 如果使用梯度权重，也对权重图进行缩放
            scaled_weight_map = F.interpolate(weight_map, size=(new_height, new_width), mode='bilinear', align_corners=False) if weight_map is not None else None
        
        # 创建对应尺度的MIND计算模块
        mind_module = MultiMIND(patch_size=patch_size, non_local_size=non_local_size, gaussian_sigma=gaussian_sigma, use_gradient_weight=use_gradient_weight)
        mind_module = mind_module.to(device)
        
        # 计算该尺度下的MIND损失
        scale_loss, _, _ = mind_module(scaled_label, scaled_eus, scaled_weight_map)
        
        # 累加加权损失
        weighted_loss = weight * scale_loss
        total_loss += weighted_loss
        
        if verbose:
            print(f"  Scale {i+1} (scale={scale}, weight={weight}): raw_loss={scale_loss.item():.6f}, weighted_loss={weighted_loss.item():.6f}")
    
    if verbose:
        print(f"  Total loss: {total_loss.item():.6f}")
    
    return total_loss


def _gaussian_blur_global(tensor, kernel_size=7, sigma=1.5):
    """
    对权重图进行高斯模糊处理
    """
    # 创建高斯核
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size).to(tensor.device).to(tensor.dtype)
    
    # 为每个通道应用高斯模糊
    batch_size, channels, height, width = tensor.shape
    tensor_reshaped = tensor.view(batch_size * channels, 1, height, width)
    blurred = F.conv2d(tensor_reshaped, kernel, padding=kernel_size//2, groups=1)
    blurred = blurred.view(batch_size, channels, height, width)
    return blurred