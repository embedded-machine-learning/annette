from __future__ import print_function

import logging
from functools import reduce
from pprint import pprint

from annette.estimation.layers.base import BaseLayer
from annette.estimation.layers.conv import ConvLayer

class ConvTransposeLayer(ConvLayer):
    """ConvTransposeLayer estimation"""

    def __init__(self, name, layer_type="ConvTranspose", est_type="roofline", op_s=1e9, bandwidth=1e9, architecture=None, y_val='ops/s'):
        super().__init__(name, layer_type, est_type, op_s, bandwidth, architecture)
        self.y_val = y_val 

    @staticmethod
    def compute_nums(layer):
        """Compute the Num Parameters for ConvTranspose Layer."""
        layer = ConvLayer.compute_nums(layer)
        return layer

    def compute_parameters(self, layer=None):
        """Compute Parameters for ConvTranspose Layer prediction"""
        self.layer = self.compute_nums(self.layer)
        return self.layer
    
    def estimate_roofline(self):
        """Returns roofline estimated ConvTranspose Layer execution time (ms)"""
        logging.debug("ConvTransposeLayer: Roofline estimation")
        logging.debug(f"Architecture: {self.architecture}")
        super().estimate_roofline()

        return self.layer['time_ms']

    def estimate_refined_roofline(self):
        """Returns roofline estimated ConvTranspose Layer execution time (ms)"""
        logging.debug("ConvTransposeLayer: Refined roofline estimation")
        super().estimate_refined_roofline()

        return self.layer['time_ms']

    def estimate_statistical(self):
        logging.debug("ConvTransposeLayer: Statistical estimation")
        super().estimate_statistical()

        return self.layer['time_ms']

    def estimate_mixed(self):
        logging.debug("ConvTransposeLayer: Mixed estimation")
        super().estimate_mixed()

        return self.layer['time_ms']
