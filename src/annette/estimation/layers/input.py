from __future__ import print_function
from pprint import pprint
from functools import reduce
import pickle
import numpy as np
import pandas as pde
from annette.estimation.layers.base import BaseLayer

class InputLayer(BaseLayer):
    """InputLayer estimation"""

    def __init__(self, name, layer_type = "Input", est_type = "roofline", op_s = 1*1e9, bandwidth = 1*1e9, architecture = None):
        self.name = name
        self.layer_type = layer_type
        self.estimation = est_type

        # Model parameters
        self.op_s = op_s 
        self.bandwidth = bandwidth 
        self.architecture = architecture

        # Layer stuff
        self.num_inputs = None
        self.num_outputs = None
        self.num_ops= None
        self.parents = None
        
        # Layer description dictionary, add information for rebuilding here
        self.desc = self.gen_dict()

    def estimate(self, layer = None):
        """return estimated InputLayer execution time (ms)"""
        print("Estimation Type: " + self.estimation)
        if hasattr(self, "estimate_" + self.estimation):
            func = getattr(self, "estimate_" + self.estimation)
            r = func(layer)
            return r
        else:
            print("No " + self.estimation + " Estimator implemented")
            return 0

    def compute_parameters(self, layer):
        """Compute Parameters for Input Layer prediction"""
        self.num_outputs = reduce(lambda x, y: x*y, layer['output_shape'][1:])
        self.num_inputs = 0
        self.num_weights = 0

        if self.architecture:
            print("noarch")


    def estimate_roofline(self, layer):
        """returns roofline estimated ConvLayer execution time (ms)"""
        print("roofline estimation")
        self.compute_parameters(layer)
        print(layer)
        #op_roof = self.num_ops / self.op_s
        if self.bandwidth == 0:
            time_ms = 0
        else:
            data_roof = (self.num_inputs + self.num_outputs) / self.bandwidth
            time_ms = np.max([data_roof])*1000 # to milliseconds
        return time_ms

    def estimate_mixed(self, layer):
        print("mixed estimation")
        self.compute_parameters(layer)
        print(layer)

    def load_estimator(self, est_model=None):
        if est_model != None:
            self.est_model = pickle.load(open(est_model, 'rb'))
            self.est_file = est_model
        else:
            self.est_model = pickle.load(open('database/conv2d_all.sav', 'rb'))