from __future__ import print_function

import logging
import pickle
from functools import reduce
from pprint import pprint

import numpy as np
from annette import get_database


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
            self.architecture = {"bit_act": 8, "bit_weights": 8}
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

    def compute_efficiency(self, unrolled, eff, div, mod, par = None, alpha = None, replication = True):
        """Compute layer efficiency for one unrolled parameter.

        Args:
            unrolled (int): unrolled parameter e.g. self.layer['output_shape'][1] for height
            eff (str): key of efficiency to write to e.g. 'h_eff'
            div (str): name of divider result. Defaults to None.
            mod (str): name of mod result. Defaults to None.
            par (str, optional): name of efficiency parameter in architecture description. Defaults to None.
            alpha (str, optional): name of the alpha parameter in the architecture description. Defaults to None.
            replication (bool, optional): replication for unrolled < par enabled. Defaults to True.
        """

        self.layer[eff] = 1
        self.layer[mod] = 1
        self.layer[div] = unrolled - 1
        if self.architecture:
            if par in self.architecture:
                if self.architecture[par] < 1:
                    self.layer[eff] = 1
                else:
                    self.layer[mod] = (self.layer[div])%self.architecture[par] + 1
                    self.layer[div] = (unrolled - self.layer[mod])/self.architecture[par]
                    if unrolled < self.architecture[par] and replication == True:
                        self.layer[eff] = 1
                    else:
                        self.layer[eff] = unrolled/(np.ceil(unrolled/self.architecture[par])*self.architecture[par])
            
                logging.debug("%s %s" % (eff, self.layer[eff]))
                if alpha in self.architecture:
                    self.layer[eff] = 1/(1-self.architecture[alpha] + 1/self.layer[eff] * (self.architecture[alpha]))
                logging.debug("%s with alpha %s" % (eff, self.layer[eff]))

    def compute_parameters(self, layer = None):
        """Compute Parameters for Base Layer prediction"""
        self.layer = self.compute_nums(self.layer)
        return self.layer

    def estimate(self, layer = None):
        """return estimated Layer execution time (ms)"""
        logging.info("Estimation Type: " + self.estimation)
        
        if hasattr(self, "estimate_" + self.estimation):
            self.layer = layer
            self.compute_parameters()
            func = getattr(self, "estimate_" + self.estimation)
            r = func()
            return r
        else:
            ("No " + self.estimation + " Estimator implemented")
            return 0

    def estimate_roofline(self):
        """returns roofline estimated BaseLayer execution time (ms)"""
        logging.info("Roofline Estimation Base Layer")
        self.layer['data_bytes'] = ((self.layer['num_outputs'] + self.layer['num_inputs']) * self.architecture['bit_act']) / 8
        self.layer['data_roof'] = self.layer['data_bytes'] / self.bandwidth
        self.layer['op_roof'] = self.layer['num_ops'] / self.op_s
        self.layer['time_ms'] = np.max([self.layer['op_roof'], self.layer['data_roof']])*1000 # to milliseconds
        if self.layer['op_roof'] > self.layer['data_roof']:
            logging.debug("OP Roof")
        else:
            logging.debug("Data Roof")
        logging.debug(self.layer['op_roof'])
        logging.debug(self.layer['data_roof'])
        logging.debug(self.layer['time_ms'])
        return self.layer['time_ms']

    def estimate_statistical(self):
        vector = self.build_vector(self.est_dict)
        result = self.est_model.predict(vector)
        self.layer['time_ms'] = self.layer['num_outputs']/result[0]*1e3

        self.layer['op_roof'] = self.layer['num_ops'] / self.op_s
        self.layer['data_roof'] = (self.layer['num_inputs'] + self.layer['num_outputs']) / self.bandwidth
        if self.layer['op_roof'] > self.layer['data_roof']:
            logging.debug("OP Roof")
        else:
            logging.debug("Data Roof")

        logging.debug(self.layer['time_ms'])
        return self.layer['time_ms']

    def estimate_mixed(self):
        return self.estimate_statistical()

    def load_estimator(self, est_model=None, est_dict=None):
        if est_model != None:
            self.est_model = pickle.load(open(get_database(est_model), 'rb'))
            self.desc['est_model'] = est_model
        else:
            return False

        self.est_dict = est_dict
        self.desc['est_dict'] = est_dict
        return True

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
