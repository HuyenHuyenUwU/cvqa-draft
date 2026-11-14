"""Ánh xạ box (xyxy chuẩn hóa) -> mặt nạ các patch trên lưới (h, w)."""
from typing import Tuple

import torch


def patch_centers(grid_size: Tuple[int, int], device) -> torch.Tensor:
    height, width = grid_size
    ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)    # [N, 2] (x, y)


def box_to_patch_mask(box_xyxy_norm: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
    """box [B,4] -> mask [B,N] float. Patch thuộc box nếu tâm patch nằm trong box; nếu không có patch nào, lấy patch gần tâm box nhất."""
    centers = patch_centers(grid_size, box_xyxy_norm.device)
    inside = ((centers[None, :, 0] >= box_xyxy_norm[:, None, 0]) & (centers[None, :, 0] <= box_xyxy_norm[:, None, 2]) &
              (centers[None, :, 1] >= box_xyxy_norm[:, None, 1]) & (centers[None, :, 1] <= box_xyxy_norm[:, None, 3]))
    box_center = torch.stack([(box_xyxy_norm[:, 0] + box_xyxy_norm[:, 2]) / 2,
                              (box_xyxy_norm[:, 1] + box_xyxy_norm[:, 3]) / 2], dim=-1)
    nearest = ((centers[None] - box_center[:, None]) ** 2).sum(-1).argmin(-1)
    empty = ~inside.any(dim=1)
    if empty.any():
        inside[empty, nearest[empty]] = True
    return inside.float()
