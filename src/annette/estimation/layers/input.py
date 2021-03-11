from __future__ import print_function

import logging
import pickle
from functools import reduce
from pprint import pprint

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
        
        if not architecture:
            self.architecture = {"bit_act": 8, "bit_weights": 8}
        else:
            self.architecture = architecture

        # Layer description dictionary, add information for rebuilding here
        self.desc = self.gen_dict()

    def estimate_roofline(self):
        """returns roofline estimated InputLayer execution time (ms)"""
        logging.info("Roofline Estimation Input Layer")
        data_bytes = ((self.layer['num_outputs']) * self.architecture['bit_act']) / 8
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
