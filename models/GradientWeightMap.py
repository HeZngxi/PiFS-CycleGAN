import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientWeightMap(nn.Module):
    """
    A PyTorch module to compute a gradient-based master weight map (W_mind)
    from a single-channel image tensor (e.g., a label map).

    The output map has values in [0, 1], where values are close to 1 at
    strong edges and close to 0 in flat regions.
    """
    def __init__(self):
        super(GradientWeightMap, self).__init__()
        # Define Sobel kernels. We register them as buffers so they will be
        # automatically moved to the correct device (CPU/GPU) with the module.
        sobel_x = torch.tensor([[-1, 0, 1], 
                                [-2, 0, 2], 
                                [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor([[1, 2, 1], 
                                [0, 0, 0], 
                                [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Calculates the master weight map.

        Args:
            image_tensor (torch.Tensor): The input image tensor, expected to be of
                                         shape [N, 1, H, W], where N is the batch size.

        Returns:
            torch.Tensor: The master weight map (W_mind) of the same shape as input,
                          with values normalized to the [0, 1] range.
        """
        # 1. Calculate Gradients using 2D convolution
        grad_x = F.conv2d(image_tensor, self.sobel_x, padding=1)
        grad_y = F.conv2d(image_tensor, self.sobel_y, padding=1)

        # 2. Calculate Gradient Magnitude
        # Add a small epsilon for numerical stability
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

        # 3. Normalize the magnitude map to [0, 1] for each image in the batch
        batch_size = grad_magnitude.shape[0]
        
        # Reshape to easily find min/max per image
        magnitude_flat = grad_magnitude.view(batch_size, -1)
        
        min_vals = torch.min(magnitude_flat, dim=1, keepdim=True)[0]
        max_vals = torch.max(magnitude_flat, dim=1, keepdim=True)[0]

        # Reshape min/max back to [N, 1, 1, 1] for broadcasting
        min_vals = min_vals.view(batch_size, 1, 1, 1)
        max_vals = max_vals.view(batch_size, 1, 1, 1)

        # Perform normalization, adding epsilon to avoid division by zero
        w_mind = (grad_magnitude - min_vals) / (max_vals - min_vals + 1e-8)
        
        return w_mind






