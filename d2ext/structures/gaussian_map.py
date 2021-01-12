# -*- coding: utf-8 -*-
# ------------------------------------------------------------
# Copyright (C) 2020-Present, Pvening, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from typing import Any, Union

import numpy as np
import torch


class HeatMap:
    def __init__(self, tensor: Union[torch.Tensor, np.ndarray]):
        pass

    def to(self, *args: Any, **kwargs: Any) -> "HeatMap":
        return HeatMap(self.tensor.to(*args, **kwargs))

    @property
    def device(self) -> torch.device:
        return self.tensor.device
