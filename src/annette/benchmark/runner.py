
from __future__ import print_function
import json
import numpy as np
import pandas as pd
import pickle as pkl
import logging
from pathlib import Path
import os

from annette import get_database 
from annette.graph import AnnetteGraph
import annette.benchmark.generator as generator
import annette.benchmark.matcher as matcher 

class Benchmark_runner():
    """Benchmark runner"""
    
    def __init__(self, network, config="config_v6.csv", match=None):
        gen = generator.Graph_generator(network)
        gen.add_configfile(config)
        gen.generate_graph_from_config(4002)

        self.match = {
                "conv2d_0_Conv2D": {
                    "conv2d_0_Relu" : "f_act",
                    "Add_0" : "f_add",
                },
                "conv2d_1_Conv2D": {
                    "conv2d_1_Relu" : "f_act",
                    "Add_0" : "f_add",
                    "Add_1" : "f_add_1",
                },
                "conv2d_2_Conv2D": {
                    "conv2d_2_Relu" : "f_act",
                    "Add_1" : "f_add_1",
                },
                "conv2d_3_Conv2D": {
                    "conv2d_3_Relu" : "f_act",
                    "concat" : "f_concat"
                },
                "conv2d_4_Conv2D": {
                    "conv2d_4_Relu" : "f_act",
                    "max_pool_MaxPool" : "f_pool",
                    "concat" : "f_concat"
                },
                "fully_conn_0_MatMul": {
                    "fully_conn_0_Relu" : "f_act",
                },
                "fully_conn_1_MatMul": {
                    "fully_conn_1_Softmax" : "f_act",
                }
            }
        match = matcher.Graph_matcher(network,config)