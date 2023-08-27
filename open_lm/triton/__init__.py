# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import torch

_triton_available = torch.cuda.is_available()
if _triton_available:
    try:
        from .layer_norm import FusedLayerNorm, layer_norm  # noqa
        from .rms_norm import FusedRmsNorm, rms_norm  # noqa

        __all__ = [
            "FusedLayerNorm",
            "layer_norm",
            "FusedRmsNorm",
            "rms_norm",
        ]
    except ImportError:
        __all__ = []