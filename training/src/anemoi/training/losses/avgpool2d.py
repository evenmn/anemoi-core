# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property

import einops
import torch
import torch.nn as nn
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.training.losses.kcrps import KernelCRPS


class AvgPool2dLoss(KernelCRPS):
    """Compute loss on convolved pred and target using average pooling.
    Intended to be used with stretched grid output, and to apply the convolution only to the high resolution part of the output.
    """

    def __init__(
        self,
        xdim: int, 
        ydim: int,
        kernel_size: list[int] = [3],
        stride: int = 1,
        no_autocast: bool = True,
        ignore_nans: bool = False,
        fair: bool = True,
        **kwargs,
    ) -> None:
        """Initialise AvgPool2dLoss.

        Parameters
        ----------
        xdim: int
            Number of grid points in x direction.
        ydim: int
            Number of grid points in y direction.
        kernel_size: int
            Size of the convolving kernel. Default is 3.
        stride: int
            Stride of the convolution. Default is 1.
        padding: int
            Padding added to all four sides of the input. Default is 1.
        """
        super().__init__(fair = fair, ignore_nans=ignore_nans, **kwargs)
        self.xdim = xdim
        self.ydim = ydim
        self.len_reg = xdim * ydim
        self.transforms = nn.ModuleList(
            nn.AvgPool2d(kernel_size=ks, stride=stride, padding=ks//2) for ks in sorted(kernel_size)
        )
        self.no_autocast = no_autocast

    

    def forward(
        self, 
        y_pred: torch.Tensor, 
        y_target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
    ) -> torch.Tensor:
        is_sharded = grid_shard_slice is not None
        assert not is_sharded, "Sharded input not supported for AvgPool2dLoss"

        bs_ = y_pred.shape[0]

        y_pred_regional = y_pred[:, :, :self.len_reg]
        y_target_regional = y_target[:, :, :self.len_reg]

        y_pred_regional = einops.rearrange(
                y_pred_regional,
                "bs e (y x) v -> (bs v) e y x",
                x=self.xdim,
                y=self.ydim,
        )
        y_target_regional = einops.rearrange(
                y_target_regional, 
                "bs (y x) v -> (bs v) y x",
                x=self.xdim,
                y=self.ydim,
        )            

        if self.no_autocast:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                loss = 0.0
                for transform in self.transforms:
                    y_pred_transformed = transform(y_pred_regional)
                    y_target_transformed = transform(y_target_regional)
                    y_pred_transformed = einops.rearrange(
                        y_pred_transformed, 
                        "(bs v) e y x -> bs e (y x) v", 
                        bs=bs_, 
                    )
                    y_target_transformed = einops.rearrange(
                        y_target_transformed, 
                        "(bs v) y x -> bs (y x) v", 
                        bs=bs_, 
                    )
                    loss += super().forward(
                        y_pred_transformed, 
                        y_target_transformed, 
                        squash=squash,
                        scaler_indices=scaler_indices,
                        without_scalers=without_scalers,
                        grid_shard_slice=None,
                        group=None,
                    )
        else:
            loss = 0.0
            for transform in self.transforms:
                y_pred_transformed = transform(y_pred_regional)
                y_target_transformed = transform(y_target_regional)
                y_pred_transformed = einops.rearrange(
                    y_pred_transformed, 
                    "(bs v) e y x -> bs e (y x) v", 
                    bs=bs_, 
                    v=y_pred.shape[-1]
                )
                y_target_transformed = einops.rearrange(
                    y_target_transformed, 
                    "(bs v) y x -> bs (y x) v", 
                    bs=bs_, 
                    v=y_target.shape[-1]
                )
                loss += super().forward(
                    y_pred_transformed, 
                    y_target_transformed, 
                    squash=squash,
                    scaler_indices=scaler_indices,
                    without_scalers=without_scalers,
                    grid_shard_slice=None,
                    group=None,
                )   

        return loss / len(self.transforms)
    
    @property
    def name(self) -> str:
        return "AvgPool2dLoss"


        


