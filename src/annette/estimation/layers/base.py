from __future__ import print_function
from pprint import pprint
from functools import reduce
import pickle
import numpy as np
import logging

class BaseLayer(object):
    """BaseLayer estimation"""

    def __init__(self, name, layer_type = "Base", est_type = "roofline", op_s = 260e9, bandwidth = 1e9, architecture = None):
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
        self.num_inputs = None
        self.num_outputs = None
        
        # Layer description dictionary, add information for rebuilding here
        self.desc = self.gen_dict()

    @staticmethod
    def compute_nums(layer):
        """Compute Num Parameters for Convolution Layer prediction"""
        try:
            layer['num_inputs'] = reduce(lambda x, y: x*y, layer['input_shape'][1:])
        except:
            layer['num_inputs'] = 0
        try:
            layer['num_outputs'] = reduce(lambda x, y: x*y, layer['output_shape'][1:])
        except:
            layer['num_outputs'] = 0
        layer['num_ops'] = 0
        return layer

    def compute_parameters(self, layer = None):
        """Compute Parameters for Base Layer prediction"""
        self.layer = self.compute_nums(self.layer)
        return self.layer

    def estimate(self, layer = None):
        """return estimated Layer execution time (ms)"""
        logging.info("Estimation Type: " + self.estimation)
        
        if hasattr(self, "estimate_" + self.estimation):
            self.layer = layer
            func = getattr(self, "estimate_" + self.estimation)
            r = func()
            return r
        else:
            ("No " + self.estimation + " Estimator implemented")
            return 0

    def estimate_roofline(self):
        """returns roofline estimated BaseLayer execution time (ms)"""
        logging.info("Roofline Estimation Base Layer")
        self.compute_parameters()
        data_bytes = ((self.layer['num_outputs'] + self.layer['num_inputs']) * self.architecture['bit_act']) / 8
        data_roof = data_bytes / self.bandwidth
        op_roof = self.layer['num_ops'] / self.op_s
        time_ms = np.max([op_roof, data_roof])*1000 # to milliseconds
        if op_roof > data_roof:
            logging.debug("OP Roof")
        else:
            logging.debug("Data Roof")
        logging.debug(op_roof)
        logging.debug(data_roof)
        logging.debug(time_ms)
        return time_ms

    def estimate_statistical(self):
        self.compute_parameters(self.layer)
        vector = self.build_vector(self.est_dict)
        result = self.est_model.predict(vector)
        time_ms = self.layer['num_outputs']/result[0]*1e3

        op_roof = self.layer['num_ops'] / self.op_s
        data_roof = (self.layer['num_inputs'] + self.layer['num_outputs']) / self.bandwidth
        if op_roof > data_roof:
            logging.debug("OP Roof")
        else:
            logging.debug("Data Roof")

        logging.debug(time_ms)
        return time_ms

    def estimate_mixed(self):
        return self.estimate_statistical()

    def load_estimator(self, est_model=None, est_dict=None):
        if est_model != None:
            self.est_model = pickle.load(open(est_model, 'rb'))
            self.desc['est_model'] = est_model
        else:
            self.est_model = pickle.load(open('database/conv2d_all.sav', 'rb'))

        self.est_dict = est_dict
        self.desc['est_dict'] = est_dict
        print(self.est_dict)

    def build_vector(self, in_vector, degree=None):
        """Build Estimation Vector"""
        vector = np.zeros([1,len(in_vector)])
        for i in in_vector.items():
            if isinstance(i[1], dict):
                vector[0,int(i[0])] = self.layer[i[1]['name']][i[1]['i']] 
                if 'dec' in i[1].keys():
                    vector[0,i[0]] = vector[0,i[0]] - i[1]['dec'] 
            elif isinstance(i[1], str):
                vector[0,int(i[0])] = self.layer[i[1]]
            else:
                vector[0,int(i[0])] = i[1]
        return vector

    def gen_dict(self, filename = None):
        desc = {"name":self.name,
                "layer_type":self.layer_type,
                "est_type":self.estimation,
                "op_s":self.op_s,
                "bandwidth":self.bandwidth,
                "architecture":self.architecture}
        return desc