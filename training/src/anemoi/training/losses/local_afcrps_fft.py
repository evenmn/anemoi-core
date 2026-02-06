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

from anemoi.training.losses.kcrps import KernelCRPS
from anemoi.training.losses.kcrps import AlmostFairKernelCRPS

LOGGER = logging.getLogger(__name__)


class Local_AFCRPSFFTLoss(AlmostFairKernelCRPS):
    """CRPS computed in spectral space, like in FourCastNet3
    """

    def __init__(
        self,
        xdim: int,
        ydim: int,
        wxdim: int = 100,
        wydim: int = 100,
        Nw: int = 10,
        alpha: float = 1.0,
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
        wxdim: int
            Shape of local window to apply transform on, x component
        wydim: int
            Shape of local window to apply transform on, y component
        Nw: int
            Number of windows to pick every time
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False
        """
        super().__init__(alpha=alpha, ignore_nans=ignore_nans, **kwargs)

        assert xdim > wxdim and ydim > wydim

        self.xdim = xdim
        self.ydim = ydim
        self.wxdim = wxdim
        self.wydim = wydim
        self.len_reg = xdim * ydim
        self.len_w = wxdim * wydim
        self.transform = torch.fft.fft2
        self.mask = self.lowpass_mask_2d(xdim, ydim)
        self.wmask = self.lowpass_mask_2d(wxdim, wydim)
        self.wrange = self.get_window_range(xdim, ydim, wxdim, wydim)
        self.no_autocast = no_autocast

    @staticmethod
    def get_window_range(xdim, ydim, wxdim, wydim, Nw):
        x1 = torch.randint(0, xdim-wxdim, Nw)
        y1 = torch.randint(0, ydim-wydim, Nw)
        x2 = x1 + wxdim
        y2 = y1 + wydim
        ranges = torch.stack([y1, y2, x1, x2]).T
        return ranges

    @staticmethod
    def lowpass_mask_2d(nx, ny):
        """
        Create a circular low-pass mask for a 2D FFT grid.

        Args:
            nx, ny: spatial sizes.
            dx, dy: sample spacing in x and y (units per pixel). Default 1.

        Returns:
            mask of shape (nx, ny), True inside the passband.
        """
        # Frequency grids in cycles per unit
        fx = torch.fft.fftfreq(nx)
        fy = torch.fft.fftfreq(ny)
        KX, KY = torch.meshgrid(fx, fy, indexing='ij')

        # Radial spatial frequency
        k = torch.sqrt(KX*KX + KY*KY)

        mask = (k < 0.5) #torch.where(k < 0.5, 1.0 - 2.0 * k, 0.0)
        mask = einops.rearrange(mask, "x y -> 1 1 (y x)")
        return mask

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

        preds_spectral = self.transform(preds)
        targets_spectral = self.transform(targets)

        preds_spectral = einops.rearrange(
                preds_spectral,
                "(bs v) e y x -> bs v (y x) e",
                bs=batch_size,
        )
        targets_spectral = einops.rearrange(
                targets_spectral,
                "(bs v) y x -> bs v (y x)",
                bs=batch_size,
        )

        kcrps_ = self._kernel_crps(preds_spectral, targets_spectral, self.alpha)
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

        # kcrps of whole field
        if self.no_autocast:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                kcrps_ = self._discrete_transform(y_pred_regional, y_target_regional, bs_)
        else:
            kcrps_ = self._discrete_transform(y_pred_regional, y_target_regional, bs_)

        kcrps_ *= self.mask.to(y_pred.device)
        kcrps_ = einops.rearrange(kcrps_, "bs v latlon -> bs 1 latlon v")
        scaled = self.scale(kcrps_, scaler_indices, without_scalers=without_scalers)

        # kcrps of windows
        pred_crop = []; target_crop = []
        for y1, y2, x1, x2 in self.wrange:
            pred_crop.append(y_pred_regional[..., y1:y2, x1:x2])
            target_crop.append(y_target_regional[..., y1:y2, x1:x2])
        pred_crop = torch.stack(pred_crop)
        target_crop = torch.stack(target_crop)
        return scaled.mean()

    @property
    def name(self) -> str:
        return "AFCRPS-FFT"
