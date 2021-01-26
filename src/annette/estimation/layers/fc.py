from __future__ import print_function
from pprint import pprint
from functools import reduce
import pickle
import numpy as np
import pandas as pd
from annette.estimation.layers.base import BaseLayer

class FullyConnectedLayer(BaseLayer):
    """FullyConnected estimation"""

    def __init__(self, name, layer_type = "Fully", est_type = "roofline", op_s = 1*1e9, bandwidth = 1*1e9, architecture = None):
        self.name = name
        self.layer_type = layer_type
        self.estimation = est_type
        self.y_val = 'ops/s'

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

        # Layer stuff
        self.layer = {}
        self.layer['num_inputs'] = None
        self.layer['num_outputs'] = None
        self.layer['num_ops'] = None
        self.layer['parents'] = None
        
        self.desc = self.gen_dict()

    def estimate(self, layer = None):
        """return estimated AdditionLayer execution time (ms)"""
        print("Estimation Type: " + self.estimation)
        if hasattr(self, "estimate_" + self.estimation):
            func = getattr(self, "estimate_" + self.estimation)
            self.layer = layer
            r = func()
            return r
        else:
            print("No " + self.estimation + " Estimator implemented")
            return 0


    @staticmethod
    def compute_nums(layer):
        """Compute Num Parameters for Layer prediction"""

        layer['num_outputs'] = reduce(lambda x, y: x*y, layer['output_shape'][1:])
        layer['num_inputs'] = reduce(lambda x, y: x*y, layer['input_shape'][1:])
        layer['num_weights'] = layer['num_inputs']*layer['num_outputs']
        layer['num_ops'] = (
            layer['num_weights']
            )*2
        
        return layer


    def compute_parameters(self):
        """Compute Parameters for Fully Connected Layer prediction"""
        self.layer = self.compute_nums(self.layer)

    def estimate_roofline(self):
        """returns roofline estimated AdditionLayer execution time (ms)"""
        print("roofline estimation")
        self.compute_parameters()
        #op_roof = self.num_ops / self.op_s
        print("Architecture: ", self.architecture)
        data_bytes = (
            (self.layer['num_outputs'] + self.layer['num_inputs']) * self.architecture['bit_act']
            + self.layer['num_weights'] * self.architecture['bit_weights']) / 8
        data_roof = data_bytes / self.bandwidth
        op_roof = self.layer['num_ops'] / self.op_s
        time_ms = np.max([op_roof, data_roof])*1000 # to milliseconds
        if op_roof > data_roof:
            print("OP Roof")
        else:
            print("Data Roof")
        print(op_roof)
        print(data_roof)
        print(time_ms)
        return time_ms

    def estimate_mixed(self):
        """returns roofline estimated Mixed execution time (ms)"""
        print("mixed estimation")
        self.compute_parameters()
        vector = self.build_vector(self.est_dict)
        print(vector)
        print(vector.shape)
        result = self.est_model.predict(vector)
        print(result)
        print(self.y_val)

        print('Operations:',self.layer['num_ops'])
        op_roof = self.layer['num_ops'] / self.op_s
        data_roof = (self.layer['num_inputs'] + self.layer['num_outputs']) / self.bandwidth
        time_ms = np.max([op_roof, data_roof])*1000 # to milliseconds
        if self.y_val == 'time(ms)':
            time_ms = result[0]
        else:
            time_ms = self.layer['num_ops']/result[0]*1e3
        if op_roof > data_roof:
            print("OP Roof")
            time_ms = time_ms
        else:
            print("Data Roof")

        if hasattr(self, 'est_model'):
            print("loaded")
        else:
            print("No estimator loaded")
        return time_ms

    def load_estimator(self, est_model=None, est_dict=None):
        if est_model != None:
            self.est_model = pickle.load(open(est_model, 'rb'))
            self.desc['est_model'] = est_model
        else:
            self.est_model = pickle.load(open('database/conv2d_all.sav', 'rb'))

        self.est_dict = est_dict
        self.desc['est_dict'] = est_dict
        print(self.est_dict)
        
    def gen_dict(self, filename = None):
        desc = {"name":self.name,
                "layer_type":self.layer_type,
                "est_type":self.estimation,
                "op_s":self.op_s,
                "bandwidth":self.bandwidth,
                "architecture":self.architecture}
        return desc