from __future__ import print_function

from pprint import pprint
from functools import reduce
import pickle
import numpy as np
import pandas as pd
import logging

from annette.estimation.layers.base import BaseLayer

class AdditionLayer(BaseLayer):
    """Addition estimation"""

    def __init__(self, name, layer_type = "Addition", est_type = "roofline", op_s = 1*1e9, bandwidth = 1*1e9, architecture = None):
        self.name = name
        self.layer_type = layer_type
        self.estimation = est_type

        # Model parameters
        self.op_s = op_s 
        self.bandwidth = bandwidth 
        if not architecture:
            self.architecture = {"bit_act": 8, "bit_weight": 8}
        else:
            self.architecture = architecture

        # Layer parameters 
        self.layer = {}
        self.layer['num_inputs'] = None
        self.layer['num_outputs'] = None
        self.layer['num_ops'] = None
        self.layer['parents'] = None
        
        # Layer description dictionary, add information for rebuilding here
        self.desc = self.gen_dict()

    def compute_parameters(self, layer = None):
        """Compute Parameters for Base Layer prediction"""
        self.layer = self.compute_nums(self.layer)
        self.layer['data_bytes'] = self.layer['num_outputs'] * self.architecture['bit_act']*3 / 8 # Element wise addition layer moves data *3
        return self.layer

    def estimate_roofline(self):
        """returns roofline estimated AdditionLayer execution time (ms)"""
        self.layer['data_roof'] = self.layer['data_bytes'] / self.bandwidth
        self.layer['time_ms'] = np.max([self.layer['data_roof']])*1000 # to milliseconds
        return self.layer['time_ms']