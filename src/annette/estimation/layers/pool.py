from __future__ import print_function
from pprint import pprint
from functools import reduce
import pickle
import numpy as np
import pandas as pde
from annette.estimation.layers.base import BaseLayer

class PoolLayer(BaseLayer):
    """PoolLayer estimation"""

    def __init__(self, name, layer_type = "Pool", est_type = "roofline", op_s = 1*1e9, bandwidth = 1*1e9, architecture = None):
        self.name = name
        self.layer_type = layer_type
        self.estimation = est_type

        # Model parameters
        self.op_s = op_s 
        self.bandwidth = bandwidth 
        self.architecture = architecture

        self.desc = self.gen_dict()

    def estimate(self, layer = None):
        """return estimated PoolLayer execution time (ms)"""
        print("Estimation Type: " + self.estimation)
        if hasattr(self, "estimate_" + self.estimation):
            func = getattr(self, "estimate_" + self.estimation)
            r = func(layer)
            return r
        else:
            print("No " + self.estimation + " Estimator implemented")
            return 0

    def compute_parameters(self, layer):
        """Compute Parameters for Pooling Layer prediction"""
        self.num_outputs = reduce(lambda x, y: x*y, layer['output_shape'][1:])
        self.num_inputs = reduce(lambda x, y: x*y, layer['output_shape'][1:2])*layer['kernel_shape'][2]*reduce(lambda x, y: x*y, layer['strides'][1:])
        if layer['pooling_type'] == 'AVG':
            self.num_ops = self.num_outputs*reduce(lambda x, y: x*y, layer['kernel_shape'][1:])*2
        else:
            self.num_ops = 0 
        print("Compute Parameters Pool:", layer)

        if self.architecture:
            print("noarch")


    def estimate_roofline(self, layer):
        """returns roofline estimated ConvLayer execution time (ms)"""
        print("roofline estimation")
        self.compute_parameters(layer)
        print(layer)
        op_roof = self.num_ops / self.op_s
        data_roof = (self.num_inputs + self.num_outputs) / self.bandwidth
        if op_roof > data_roof:
            print("OP Roof")
        else:
            print("Data Roof")
        time_ms = np.max([data_roof, op_roof])*1000 # to milliseconds
        print(op_roof)
        print(data_roof)
        print(time_ms)
        return time_ms