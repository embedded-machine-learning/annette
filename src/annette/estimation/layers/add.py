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
        if architecture:
            self.architecture = architecture 
        else:
            self.architecture = {}
        if not "bit_act" in self.architecture:
            self.architecture["bit_act"] = 8
        if not "bit_weights" in self.architecture:
            self.architecture["bit_weights"] = 8

        # Layer parameters 
        self.layer = {}
        self.layer['num_inputs'] = None
        self.layer['num_outputs'] = None
        self.layer['num_ops'] = None
        self.layer['parents'] = None
        
        self.desc = self.gen_dict()

    def estimate_roofline(self):
        """returns roofline estimated AdditionLayer execution time (ms)"""
        self.compute_parameters()
        data_bytes = self.layer['num_outputs'] * self.architecture['bit_act']*3 / 8 # Element wise addition layer moves data *3
        data_roof = data_bytes / self.bandwidth
        time_ms = np.max([data_roof])*1000 # to milliseconds
        print(data_roof)
        return time_ms

    def estimate_mixed(self):
        """returns roofline estimated Mixed execution time (ms)"""
        self.compute_parameters()
        print(self.layer)
        if hasattr(self, 'est_model'):
            print("loaded")
        else:
            print("No estimator loaded")
            return 0

    def gen_dict(self, filename = None):
        desc = {"name":self.name,
                "layer_type":self.layer_type,
                "est_type":self.estimation,
                "op_s":self.op_s,
                "bandwidth":self.bandwidth,
                "architecture":self.architecture}
        return desc