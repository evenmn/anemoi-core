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
import ptwt #torch wavelets
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.training.losses.kcrps import AlmostFairKernelCRPS #KernelCRPS

LOGGER = logging.getLogger(__name__)


class CRPSWaveletLoss2(AlmostFairKernelCRPS):
    """CRPS aggregated across multiple scales using a wavelet transform.
    """

    def __init__(
        self,
        xdim: int,
        ydim: int,
        cutoff_ratio: float = 1.0,
        alpha: float = 1.0,
        maxlevel: int = 9,
        include_reverse: bool = False,
        include_smooth: bool = False,
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
        alpha: float, optional
            Weighting between fair and almost fair, by default 1.0
        maxlevel: int, optional
            Number of levels in the wavelet transform, by default 3
        include_reverse: bool, optional
            Whether to include the reverse coefficients, starting with 'dhv' and smoothing after.
        include_smooth: bool, optional
            Include the smooth coefficients when computing the loss.
        eps: float, optional
            Normalizing factor for transformed field for numerical stability
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False
        """
        super().__init__(ignore_nans=ignore_nans, **kwargs)

        self.xdim = xdim
        self.ydim = ydim
        self.len_reg = xdim * ydim
        self.transform = ptwt.WaveletPacket2D
        self.maxlevel = maxlevel
        self.no_autocast = no_autocast

        self.select_coeffs = []
        end_coeffs = ['d','h','v'] if not include_smooth else ['a','d','h','v']
        for i in range(self.maxlevel):
            self.select_coeffs += [ 'a'*i + c for c in end_coeffs]
            if i >= 1 and include_reverse:
                self.select_coeffs += [c + 'a'*i for c in end_coeffs]

    def _discrete_transform(self, preds: torch.Tensor, targets: torch.Tensor, batch_size: int, var_count: int) -> torch.Tensor:
        """
        Calculates the tree of discrete wavelet transforms at maxlevel levels.
        The fields are flattened in (y,x) and concatanated along this axis.
        Crps is then calculated on the total flattened array.

        Args:
            preds: torch.Tensor
                Predictions, (bs*var, ens, y, x)
            targets: torch.Tensor
                Targets, (bs*var, y, x)
            batch_size: int
                Self-explanatory
            var_count: int
                Number of variables in the batch
        """

        wpred = self.transform(preds, "haar", maxlevel=self.maxlevel, axes = (-2, -1), mode = 'constant')
        wtarget = self.transform(targets, "haar", maxlevel=self.maxlevel, axes = (-2, -1), mode = 'constant')

        #Calculate grid space shapes of different levels
        shapes = [wtarget['a'*i].shape[-2:] for i in range(1,self.maxlevel+1)]

        kcrps = 0
        kc = 0

        for c in self.select_coeffs:  #all_coeffs:
            #norm = 2**(len(c))  #normalize for number of levels.
            cpred = einops.rearrange(wpred[c], "(bs v) e y x -> bs v (y x) e", bs=batch_size) #/norm
            ctarget = einops.rearrange(wtarget[c], "(bs v) y x -> bs v (y x)", bs=batch_size) #/norm
            
            norm = torch.std(ctarget, dim=-1, keepdim=True) + 1e-4 # could technically be precomputed per variable.
            ctarget = ctarget / norm
            cpred = cpred / norm[..., None]

            kcrps_ = self._kernel_crps(cpred, ctarget, alpha = self.alpha)
            #Average over the grid points in this coefficient, but add singleton dimension in its place
            #kcrps_ currently has shape (bs, v, y*x)
            #This is fine because we are not doing any scaling on the grid dimension. And ensures consistent shape between coeffs.
            kcrps_ = kcrps_.mean(dim=-1, keepdim=True)
            kcrps += kcrps_
            kc += 1
            #print(f"Computed CRPS for coeff {c}, shape {cpred.shape}, current kcrps {kcrps_.mean().item():.6f}")

        #print("Total kcrps:", kcrps.mean().item(), "over", kc, "coefficients.")
        #print("Average kcrps:", (kcrps/kc).mean().item())

        return kcrps / kc  #average over number of coeffs


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
        v_ = y_pred.shape[-1]  # number of variables

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
                kcrps_ = self._discrete_transform(y_pred_regional, y_target_regional, batch_size=bs_, var_count=v_)
        else:
            kcrps_ = self._discrete_transform(y_pred_regional, y_target_regional, batch_size=bs_, var_count=v_)

        #torch.save(log_diff, "/leonardo/home/userexternal/enordhag/log_diff.pt")

        kcrps_ = einops.rearrange(kcrps_, "bs v latlon -> bs 1 latlon v")
        scaled = self.scale(kcrps_, scaler_indices, without_scalers=without_scalers)
        #print("FFT loss:", scaled.mean())
        return scaled.mean() #self.reduce(kcrps_, squash=squash, squash_mode="avg", group=None)

    @property
    def name(self) -> str:
        return f"CRPS-Waveletv2{self.alpha:.2f}"