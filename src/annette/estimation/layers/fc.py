from __future__ import print_function
from pprint import pprint
from functools import reduce
import numpy as np
import logging
from annette.estimation.layers.base import BaseLayer

class FullyConnectedLayer(BaseLayer):
    """FullyConnected estimation"""

    def __init__(self, name, layer_type="FullyConnected", est_type="roofline", op_s=1e9, bandwidth=1e9, architecture=None, y_val='ops/s'):
        super().__init__(name, layer_type, est_type, op_s, bandwidth, architecture)
        self.y_val = y_val

    @staticmethod
    def compute_nums(layer):
        """Compute Num Parameters for Layer prediction"""
        layer = BaseLayer.compute_nums(layer)
        layer['num_weights'] = layer['num_inputs'] * layer['num_outputs']
        layer['num_ops'] = layer['num_weights'] * 2
    
        return layer

    def compute_parameters(self):
        """Compute Parameters for FullyConnected Layer prediction"""
        self.layer = self.compute_nums(self.layer)
        return self.layer

    def compute_eff(self):
        self.compute_efficiency(
            self.layer['input_shape'][1], 'c_eff', 'c_div', 'c_mod', 'c_par', 'c_alpha',
            replication=True
        )
        self.compute_efficiency(
            self.layer['output_shape'][1], 'f_eff', 'f_div', 'f_mod', 'f_par', 'f_alpha',
            replication=True
        )
        self.layer['eff'] = (
            self.layer['c_eff'] * self.layer['f_eff']
        )

        return self.layer

    def estimate_roofline(self):
        """Returns roofline estimated FullyConnected layer execution time (ms)"""
        logging.debug("FullyConnected layer: Roofline estimation")
        logging.debug(f"Architecture: {self.architecture}")
        super().estimate_roofline()

        return self.layer['time_ms']

    def estimate_refined_roofline(self):
        """Returns roofline estimated FullyConnected layer execution time (ms)"""
        logging.debug("FullyConnected layer: Refined roofline estimation")
        super().estimate_refined_roofline()

        return self.layer['time_ms']

    def estimate_statistical(self):
        logging.debug("FullyConnected layer: Statistical estimation")
        super().estimate_statistical()

        return self.layer['time_ms']

    def estimate_mixed(self):
        logging.debug("FullyConnected layer: Mixed estimation")
        super().estimate_mixed()

        return self.layer['time_ms']
