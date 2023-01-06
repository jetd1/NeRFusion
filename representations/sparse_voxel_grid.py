import torch
from torch import nn
from torchsparse import PointTensor, SparseTensor
from torchsparse.utils.quantize import sparse_quantize


# https://github.com/zju3dv/NeuralRecon/blob/master/ops/generate_grids.py
def generate_grid(n_vox, interval):
    with torch.no_grad():
        # Create voxel grid
        grid_range = [torch.arange(0, n_vox[axis], interval) for axis in range(3)]
        grid = torch.stack(torch.meshgrid(grid_range[0], grid_range[1], grid_range[2]))  # 3 dx dy dz
        grid = grid.unsqueeze(0).cuda().float()  # 1 3 dx dy dz
        grid = grid.view(1, 3, -1)
    return grid


class SparseVoxelGrid(nn.Module):
    def __init__(self, scale, resolution, feat_dim):
        """
        scale: range of xyz. 0.5 -> (-0.5, 0.5)
        resolution: #voxels within each dim. 128 -> 128x128x128
        """
        super().__init__()

        self.scale = scale
        self.resolution = resolution
        self.voxel_size = scale * 2 / resolution




