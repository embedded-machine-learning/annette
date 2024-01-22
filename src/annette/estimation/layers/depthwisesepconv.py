from __future__ import print_function
from pprint import pprint
from functools import reduce
import logging
import numpy as np
from annette.estimation.layers.base import BaseLayer
from annette.estimation.layers.conv import ConvLayer
from annette.estimation.layers.depthwiseconv import DepthwiseConvLayer

class DepthwiseSepConvLayer(DepthwiseConvLayer):
    """DepthwiseSepConvLayer estimation"""

    def __init__(self, name, layer_type="DepthwiseSepConv", est_type="mixed", op_s=1e9, bandwidth=1e9, architecture=None, y_val='ops/s'):
        super().__init__(name, layer_type, est_type, op_s, bandwidth, architecture)
        self.y_val = y_val

    @staticmethod
    def compute_nums(layer):
        """Compute Num Parameters for Depthwise Separable Convolution Layer prediction"""
        layer = DepthwiseConvLayer.compute_nums(layer)
        return layer

    def estimate_roofline(self):
        """Returns roofline estimated DepthwiseSeparableConvLayer execution time (ms)"""
        logging.debug("DepthwiseSeparableConvLayer: Roofline estimation")
        logging.debug(f"Architecture: {self.architecture}")
        super().estimate_roofline()

        return self.layer['time_ms']

    def estimate_refined_roofline(self):
        """Returns roofline estimated DepthwiseSeparableConvLayer execution time (ms)"""
        logging.debug("DepthwiseSeparableConvLayer: Refined roofline estimation")
        super().estimate_refined_roofline()

        return self.layer['time_ms']

    def estimate_statistical(self):
        logging.debug("DepthwiseSeparableConvLayer: Statistical estimation")
        super().estimate_statistical()

        return self.layer['time_ms']

    def estimate_mixed(self):
        logging.debug("DepthwiseSeparableConvLayer: Mixed estimation")
        super().estimate_mixed()

        # DwConv execution time seems to be almost halved in networks with multiple layers 
        # in aiw for k = 3. Could also be a bug!?
        if self.layer['kernel_shape'][0] == 3 and self.layer['kernel_shape'][1] == 3:
            return self.layer['time_ms'] * 0.535

        return self.layer['time_ms']
