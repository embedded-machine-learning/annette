from __future__ import print_function

from pprint import pprint
from functools import reduce
import logging

from annette.estimation.layers.base import BaseLayer

class ConvLayer(BaseLayer):
    """ConvLayer estimation"""

    def __init__(self, name, layer_type="Conv", est_type="roofline", op_s=1e9, bandwidth=1e9, architecture=None, y_val='ops/s'):
        super().__init__(name, layer_type, est_type, op_s, bandwidth, architecture, y_val)
        self.y_val = y_val
        self.layer['y_val'] = y_val

    @staticmethod
    def compute_nums(layer):
        """Compute Num Parameters for Convolution Layer prediction"""
        layer = BaseLayer.compute_nums(layer)
        layer['num_weights'] = reduce(lambda x, y: x*y, layer['kernel_shape'])
        layer['num_ops'] = (
            layer['num_weights']
            * layer['output_shape'][1] * layer['output_shape'][2]
            * 2  # 1 MAC = 2 ops
        )
        
        return layer

    def compute_parameters(self, layer=None):
        """Compute Parameters for Convolution Layer prediction"""
        self.layer = self.compute_nums(self.layer)
        return self.layer

    def compute_eff(self):
        self.compute_efficiency(
            self.layer['output_shape'][2], 'h_eff', 'h_div', 'h_mod', 'h_par', 'h_alpha',
            replication=True
        )
        self.compute_efficiency(
            self.layer['output_shape'][1], 'w_eff', 'w_div', 'w_mod', 'w_par', 'w_alpha',
            replication=True
        )
        self.compute_efficiency(
            self.layer['kernel_shape'][2], 'c_eff', 'c_div', 'c_mod', 'c_par', 'c_alpha',
            replication=True
        )
        self.compute_efficiency(
            self.layer['kernel_shape'][3], 'f_eff', 'f_div', 'f_mod', 'f_par', 'f_alpha',
            replication=True
        )
        self.layer['eff'] = (
            self.layer['h_eff'] * self.layer['w_eff']
            * self.layer['c_eff'] * self.layer['f_eff']
        )

        return self.layer

    def estimate_roofline(self):
        """Returns roofline estimated ConvLayer execution time (ms)"""
        logging.debug("ConvLayer: Roofline estimation")
        logging.debug(f"Architecture: {self.architecture}")
        super().estimate_roofline()

        return self.layer['time_ms']

    def estimate_refined_roofline(self):
        """Returns roofline estimated ConvLayer execution time (ms)"""
        logging.debug("ConvLayer: Refined roofline estimation")
        super().estimate_refined_roofline()

        return self.layer['time_ms']

    def estimate_statistical(self):
        logging.debug("ConvLayer: Statistical estimation")
        super().estimate_statistical()

        return self.layer['time_ms']

    def estimate_mixed(self):
        logging.debug("ConvLayer: Mixed estimation")
        super().estimate_mixed()

        return self.layer['time_ms']
