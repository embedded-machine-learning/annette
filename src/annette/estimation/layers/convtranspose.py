from __future__ import print_function

import logging
import pickle
from functools import reduce
from pprint import pprint

import numpy as np
import pandas as pd
from annette.estimation.layers.base import BaseLayer

class ConvTransposeLayer(BaseLayer):
    """ConvTransposeLayer estimation"""

    def __init__(self, name, layer_type = "ConvTranspose", est_type = "roofline", op_s = 1*1e9, bandwidth = 1*1e9, architecture = None):
        self.name = name
        self.layer_type = layer_type
        self.estimation = est_type

        # Model parameters
        self.op_s = op_s 
        self.bandwidth = bandwidth 
        if not architecture:
            self.architecture = {"bit_act": 8, "bit_weights": 8}
        else:
            self.architecture = architecture
        self.y_val = 'ops/s'

        # Layer description dictionary, add information for rebuilding here
        self.desc = self.gen_dict()

    @staticmethod
    def compute_nums(layer):
        """Compute the Num Parameters for ConvolutionTranspose Layer.

        Args:
            layer ([type]): [description]

        Returns:
            [type]: [description]
        """

        layer['num_weights'] = reduce(lambda x, y: x*y, layer['kernel_shape'])
        layer['num_outputs'] = reduce(lambda x, y: x*y, layer['output_shape'][1:])
        layer['num_inputs'] = reduce(lambda x, y: x*y, layer['input_shape'][1:])
        layer['num_ops'] = (
            layer['num_weights'] * layer['output_shape'][1] * layer['output_shape'][2]
            )*2
        
        return layer

    def compute_parameters(self, layer = None):
        """Compute Parameters for Convolution Layer prediction"""

        self.layer = self.compute_nums(self.layer)

        self.compute_efficiency(self.layer['output_shape'][1], 'h_eff', 'h_div', 'h_mod', 'h_par', 'h_alpha')
        self.compute_efficiency(self.layer['output_shape'][2], 'w_eff', 'w_div', 'w_mod', 'w_par', 'w_alpha')
        self.compute_efficiency(self.layer['kernel_shape'][2], 'c_eff', 'c_div', 'c_mod', 'c_par', 'c_alpha')
        self.compute_efficiency(self.layer['kernel_shape'][3], 'f_eff', 'f_div', 'f_mod', 'f_par', 'f_alpha')
        self.layer['eff'] = self.layer['h_eff'] *self.layer['w_eff']*self.layer['c_eff']*self.layer['f_eff']

        return self.layer

    def estimate_roofline(self):
        """returns roofline estimated ConvLayer execution time (ms)"""
        logging.debug("roofline estimation")
        self.layer['op_roof'] = self.layer['num_ops'] / self.op_s
        self.layer['data_bytes'] = (
            (self.layer['num_inputs'] + self.layer['num_outputs'] )* self.architecture['bit_act']
            + self.layer['num_weights'] * self.architecture['bit_weights']) / 8
        logging.debug("Architecture: %s" % self.architecture)
        self.layer['data_roof'] = self.layer['data_bytes'] / self.bandwidth

        self.layer['time_ms'] = np.max([self.layer['op_roof'], self.layer['data_roof']])*1000 # to milliseconds
        if self.layer['op_roof'] > self.layer['data_roof']:
            logging.debug("OP Roof")
        else:
            logging.debug("Data Roof")
        return self.layer['time_ms']

    def estimate_refined_roofline(self):
        """returns roofline estimated ConvLayer execution time (ms)"""
        logging.debug("refined roofline estimation")
        self.estimate_roofline()

        if self.layer['op_roof'] > self.layer['data_roof']:
            logging.debug("OP Roof")
            logging.debug(self.layer['eff'])
            self.layer['time_ms'] = self.layer['time_ms'] / self.layer['eff']
        else:
            logging.debug("Data Roof")
        logging.debug(self.layer['op_roof'])
        logging.debug(self.layer['data_roof'])
        logging.debug(self.layer['time_ms'])
        return self.layer['time_ms']

    def estimate_statistical(self):
        logging.debug("statistical estimation")
        vector = self.build_vector(self.est_dict)
        result = self.est_model.predict(vector)
        logging.debug(result)

        self.layer['op_roof'] = self.layer['num_ops'] / self.op_s
        self.layer['data_roof'] = (self.layer['num_inputs'] + self.layer['num_outputs']) / self.bandwidth
        self.layer['stat_est'] = self.layer['num_ops']/result[0]*1e3

        if self.layer['op_roof'] > self.layer['data_roof']:
            logging.debug("OP Roof")
            self.layer['time_ms'] = self.layer['op_roof']
        else:
            logging.debug("Data Roof")
            self.layer['time_ms'] = self.layer['data_roof']

        self.layer['time_ms'] = self.layer['stat_est']

        logging.debug(self.layer['time_ms'])
        return self.layer['time_ms']

    def estimate_mixed(self):
        logging.debug("mixed estimation")
        self.estimate_statistical()

        logging.debug('Operations: %i' % self.layer['num_ops'])

        if self.layer['op_roof'] > self.layer['data_roof']:
            logging.debug("OP Roof")
            self.layer['time_ms'] = self.layer['time_ms'] / (self.layer['h_eff']*self.layer['c_eff']*self.layer['f_eff'])
        else:
            logging.debug("Data Roof")

        return self.layer['time_ms']