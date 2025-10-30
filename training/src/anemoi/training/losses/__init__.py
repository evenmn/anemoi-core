# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from .combined import CombinedLoss
from .huber import HuberLoss
from .kcrps import AlmostFairKernelCRPS
from .kcrps import KernelCRPS
#from .dct import DCTLoss
#from .crps_fft import CRPSFFTLoss
#from .crps_fftw import CRPSFFTWLoss
#from .crps_wavelet import CRPSWaveletLoss
from .crps_wave2 import CRPSWaveletLoss2
from .logcosh import LogCoshLoss
from .loss import get_loss_function
from .mae import MAELoss
from .mse import MSELoss
from .rmse import RMSELoss
from .weighted_mse import WeightedMSELoss

__all__ = [
    "AlmostFairKernelCRPS",
    "CombinedLoss",
    "HuberLoss",
    "KernelCRPS",
#    "DCTLoss",
 #   "CRPSFFTLoss",
 #   "CRPSFFTWLoss",
 #   "CRPSWaveletLoss",
    "CRPSWaveletLoss2",
    "LogCoshLoss",
    "MAELoss",
    "MSELoss",
    "RMSELoss",
    "WeightedMSELoss",
    "get_loss_function",
]
