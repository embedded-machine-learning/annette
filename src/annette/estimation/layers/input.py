from __future__ import print_function

import logging
from functools import reduce
from pprint import pprint

from annette.estimation.layers.base import BaseLayer


class InputLayer(BaseLayer):
    """InputLayer estimation"""

    def __init__(self, name, layer_type="DataInput", est_type="roofline", op_s=1e9, bandwidth=1e9, architecture=None, y_val='ops/s'):
        super().__init__(name, layer_type, est_type, op_s, bandwidth, architecture)

    @staticmethod
    def compute_nums(layer):
        layer = BaseLayer.compute_nums(layer)
        layer['num_inputs'] = 0

        return layer

    def estimate_roofline(self):
        super().estimate_roofline()
        return self.layer['time_ms']

    def estimate_refined_roofline(self):
        return self.estimate_roofline()

    def estimate_statistical(self):
        return self.estimate_roofline()

    def estimate_mixed(self):
        return self.estimate_roofline()
