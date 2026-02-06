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
from torch.distributed.distributed_c10d import ProcessGroup
from pytorch_wavelets import DWTForward

from anemoi.training.losses.kcrps import KernelCRPS
from anemoi.training.losses.kcrps import AlmostFairKernelCRPS

LOGGER = logging.getLogger(__name__)


class AFCRPSWTLoss(AlmostFairKernelCRPS):
    """Spectral CRPS loss with wavelets
    """

    def __init__(
        self,
        xdim: int,
        ydim: int,
        alpha: float = 1.0,
        levels: int = 5,
        wave: str = "db4",
        no_autocast: bool = True,
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        """Latitude- and (inverse-)variance-weighted kernel CRPS loss.

        Parameters
        ----------
        xdim: int
            Shape of regional domain to apply transform on, x component
        ydim: int
            Shape of regional domain to apply transform on, y component
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False
        """
        super().__init__(alpha=alpha, ignore_nans=ignore_nans, **kwargs)

        self.xdim = xdim
        self.ydim = ydim
        self.len_reg = xdim * ydim
        # Note: to avoid inverse transform, we apply forward transform levels time with J=1
        self.dwt = DWTForward(J=1, wave=wave, mode="zero")
        self.levels = levels
        self.no_autocast = no_autocast


    def _discrete_transform(self, preds: torch.Tensor, targets: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Perform the discrete Fourier/cosine transform of preds and targets and return log-diff.

        Args:
            preds: torch.Tensor
                Predictions, (bs*var, ens, y, x)
            targets: torch.Tensor
                Targets, (bs*var, y, x)
            batch_size: int
                Self-explanatory
        """

        preds_l = preds
        targets_l = targets.unsqueeze(0)

        # initialize kcrps
        bsvar, y, x = targets.shape
        var = bsvar // batch_size
        kcrps_ = torch.zeros(batch_size, var).to(preds.device)

        for _ in range(self.levels):
            #print("preds_l.shape:", preds_l.shape)
            #print("targets_l.shape:", targets_l.shape)
            preds_l, _ = self.dwt(preds_l)
            targets_l, _ = self.dwt(targets_l)

            #print("preds_l.shape:", preds_l.shape)
            #print("targets_l.shape:", targets_l.shape)

            sigma_targets = torch.var(targets_l, dim=(-2, -1), unbiased=False).unsqueeze(-1)
            #print("sigma_targets.shape:", sigma_targets.shape)

            preds_spectral = einops.rearrange(
                    preds_l,
                    "(bs v) e y x -> bs v (y x) e",
                    bs=batch_size,
            )
            targets_spectral = einops.rearrange(
                    targets_l,
                    "1 (bs v) y x -> bs v (y x)",
                    bs=batch_size,
            )

            #print(preds_spectral.shape)
            #print(targets_spectral.shape)

            kcrps_tmp = self._kernel_crps(preds_spectral, targets_spectral, self.alpha) / sigma_targets
            #print(kcrps_tmp.shape)
            kcrps_ += kcrps_tmp.mean(dim=-1)
        return kcrps_


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
        assert not is_sharded, "Set 'keep_batch_sharded=False' in the model config to compute spectral loss"

        bs_ = y_pred.shape[0]  # batch size

        # Reshape to 2D grid
        y_pred_regional = y_pred[:, :, :self.len_reg]
        y_target_regional = y_target[:, :self.len_reg]
        
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
                kcrps_ = self._discrete_transform(y_pred_regional, y_target_regional, bs_)
        else:
            kcrps_ = self._discrete_transform(y_pred_regional, y_target_regional, bs_)

        kcrps_ = einops.rearrange(kcrps_, "bs v -> bs 1 1 v")
        scaled = self.scale(kcrps_, scaler_indices, without_scalers=without_scalers)
        print("AFCRPSWT loss:", scaled.mean())
        return scaled.mean()

    @property
    def name(self) -> str:
        return "AFCRPS-Wavelet"
