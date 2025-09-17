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
from torch_dct import dct_2d
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.training.losses.base import BaseLoss

LOGGER = logging.getLogger(__name__)


class DCTLoss(BaseLoss):
    """Relative error of the discrete cosine transform (DCT) of the fields.
    This is similar to FFT2, but without boundary problems associated with FFT.

    Ref.: IEEE Transactions on Acoustics, Speech, and Signal Processing ( Volume: 28, Issue: 1, February 1980) 
    """

    def __init__(
        self,
        xdim: int,
        ydim: int,
        cutoff_ratio: float = 1.0,
        fft: bool = False,
        alpha: float = 1.0,  # remove this
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
        skip_last: int, optional
            Skip the end points of the spectrum due to artifacts
        fft: bool, optional
            Do the Fourier transform instead of discrete cosine, by default False
        eps: float, optional
            Normalizing factor for transformed field for numerical stability
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False
        """
        super().__init__(ignore_nans=ignore_nans, **kwargs)

        self.xdim = xdim
        self.ydim = ydim
        self.len_reg = xdim * ydim
        self.transform = torch.fft.fft2 if fft else dct_2d
        self.mask = self.lowpass_mask_2d(xdim, ydim, cutoff_ratio)
        self.no_autocast = no_autocast

    @staticmethod
    def lowpass_mask_2d(nx, ny, cutoff_ratio=1.0, *,
                        dx=1.0, dy=1.0,
                        shifted=False):
        """
        Create a circular low-pass mask for a 2D FFT grid.

        Args:
            nx, ny: spatial sizes.
            cutoff_ratio: 0..1, relative to the radial Nyquist (1 is as large as allowed).
            dx, dy: sample spacing in x and y (units per pixel). Default 1.
            shifted: if True, mask is centered (for use with fftshifted spectra).
                     if False, mask matches unshifted torch.fft.fft2 output.

        Returns:
            mask of shape (nx, ny), True inside the passband.
        """
        # Frequency grids in cycles per unit
        fx = torch.fft.fftfreq(nx, d=dx)
        fy = torch.fft.fftfreq(ny, d=dy)
        KX, KY = torch.meshgrid(fx, fy, indexing='ij')

        # Radial spatial frequency
        k = torch.sqrt(KX*KX + KY*KY)

        # Convert ratio -> absolute cutoff in cycles per unit
        fx_nyq = 1.0 / (2.0 * dx)
        fy_nyq = 1.0 / (2.0 * dy)
        k_nyq_radial = min(fx_nyq, fy_nyq)  # circular limit
        k_cut = float(cutoff_ratio) * k_nyq_radial

        mask = (k <= k_cut)

        # Align to shifted spectrum if requested
        if shifted:
            mask = torch.fft.fftshift(mask, dim=(-2, -1))

        return mask.T

    def _discrete_transform(self, preds: torch.Tensor, targets: torch.Tensor, one_member: bool = False, eps: float = 1e-8) -> torch.Tensor:
        """
        Perform the discrete Fourier/cosine transform of preds and targets and return log-diff.

        Args:
            preds: torch.Tensor
                Predictions, (bs*var, ens, y, x)
            targets: torch.Tensor
                Targets, (bs*var, y, x)
            one_member: bool
                Restrict loss computation to one member to save compute and memory, False by default
            eps: float
                Add this (small) value to avoid zero division, 1e-8 by default
        """

        preds_spectral = self.transform(preds) + eps
        targets_spectral = self.transform(targets) + eps

        if one_member:
            preds_spectral = preds_spectral[:, 0]
        else:
            targets_spectral = targets_spectral.unsqueeze(1)

        log_diff = torch.abs(torch.log(preds_spectral / targets_spectral))
        log_diff *= self.mask.to(log_diff.device)

        return log_diff


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
        assert not is_sharded, "Set 'keep_batch_sharded=False in the model config to compute spectral loss"

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

        if self.no_autocast:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                log_diff = self._discrete_transform(y_pred_regional, y_target_regional)
        else:
            log_diff = self._discrete_transform(y_pred_regional, y_target_regional)

        log_diff = einops.rearrange(
                log_diff, 
                "(bs v) e y x -> bs e (y x) v",
                bs=bs_,
        )
        scaled = self.scale(log_diff, scaler_indices, without_scalers=without_scalers)

        # divide by (weighted point count) * (batch size)
        if squash:
            return scaled.mean() / bs_

        loss = scaled.mean(dim=0) / bs_
        return loss.mean(dim=0)

    @property
    def name(self) -> str:
        return "dct"
