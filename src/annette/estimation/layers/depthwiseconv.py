from __future__ import print_function
from pprint import pprint
from functools import reduce
import logging
import numpy as np
from annette.estimation.layers.base import BaseLayer
from annette.estimation.layers.conv import ConvLayer

class DepthwiseConvLayer(ConvLayer):
    """DepthwiseConvLayer estimation"""

    def __init__(self, name, layer_type="DepthwiseConv", est_type="roofline", op_s=1e9, bandwidth=1e9, architecture=None, y_val='ops/s'):
        super().__init__(name, layer_type, est_type, op_s, bandwidth, architecture)
        self.y_val = y_val

    @staticmethod
    def compute_nums(layer):
        """Compute Num Parameters for Depthwise Convolution Layer prediction"""
        layer = ConvLayer.compute_nums(layer)
        return layer

    def estimate_roofline(self):
        """Returns roofline estimated DepthwiseConvLayer execution time (ms)"""
        logging.debug("DepthwiseConvLayer: Roofline estimation")
        logging.debug(f"Architecture: {self.architecture}")
        super().estimate_roofline()

        return self.layer['time_ms']

    def estimate_refined_roofline(self):
        """Returns roofline estimated DepthwiseConvLayer execution time (ms)"""
        logging.debug("DepthwiseConvLayer: Refined roofline estimation")
        super().estimate_refined_roofline()

        return self.layer['time_ms']

    def estimate_statistical(self):
        logging.debug("DepthwiseConvLayer: Statistical estimation")
        super().estimate_statistical()

        return self.layer['time_ms']

    def estimate_mixed(self):
        logging.debug("DepthwiseConvLayer: Mixed estimation")
        super().estimate_mixed()

        return self.layer['time_ms']
