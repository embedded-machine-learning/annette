from __future__ import print_function

from pprint import pprint
from functools import reduce
import numpy as np
import logging

from annette.estimation.layers.base import BaseLayer

class AdditionLayer(BaseLayer):
    """Addition estimation"""

    def __init__(self, name, layer_type="Add", est_type="roofline", op_s=1e9, bandwidth=1e9, architecture=None, y_val='ops/s'):
        super().__init__(name, layer_type, est_type, op_s, bandwidth, architecture, y_val=y_val)

    @staticmethod
    def compute_nums(layer):
        layer = BaseLayer.compute_nums(layer)
        layer['num_inputs'] *= 2  # 2 input tensors of same size
        layer['num_ops'] = layer['num_outputs']  # 1 addition per output element

        return layer

    def compute_parameters(self, layer=None):
        """Compute Parameters for AdditionLayer prediction"""
        self.layer = self.compute_nums(self.layer)
        return self.layer

    def estimate_roofline(self):
        """Returns roofline estimated AddLayer execution time (ms)"""
        logging.debug("AddLayer: Roofline estimation")
        super().estimate_roofline()

        return self.layer['time_ms']

    def estimate_refined_roofline(self):
        return self.estimate_roofline()

    def estimate_statistical(self):
        return self.estimate_roofline()

    def estimate_mixed(self):
        return self.estimate_roofline()
