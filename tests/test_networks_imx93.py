# %%
import logging
import os
import pickle
import pathlib
import sys

from annette.estimate import estimate
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from annette.utils import get_database
from pprint import pprint
from annette.graph.graph_util.annette_graph import AnnetteGraph
import annette.benchmark.matcher as matcher
import annette.hw_modules.hw_modules.ncs2 as ncs2
import annette.hw_modules.hw_modules.imx93 as imx93 
import annette.hw_modules.hw_modules.imx93 as imx93
from annette.estimation.layer_model import Layer_model
from annette.estimation.mapping_model import Mapping_model



#logging.basicConfig(level=logging.DEBUG)

def main():

    test = True
    visualize = True 

    #networks=["cf_openpose", "cf_reid", "cf_landmark", "cf_inceptionv1", "cf_inceptionv2", "cf_inceptionv3", "cf_inceptionv4", "cf_squeezenet"]
    #networks = ["pt_resnet50", "cf_openpose", "cf_inceptionv3"]
    networks = ["cf_inceptionv3"]
    config="dummy.csv"
    layer = 'imx93-stat'
    #layer = 'ov_cpu_mixed'

    #mapping = 'ov2'
    mapping = 'simple'

    device = 'MYRIAD'
    #device = 'CPU'
    device = 'imx93'

    network = networks[0]

    imx93_obj = imx93.inference.imx93Class(get_database('configs','imx93.yaml'))
    graph_matcher = matcher.Graph_matcher(network, config)
    result = graph_matcher.run_bench(optimize = imx93_obj.optimize_network, execute = imx93_obj.run_network_ssh, parse = imx93.parser.r2a, hardware='imx93',
        execute_kwargs = {"model_path" : get_database('benchmarks', 'tmp'), 'sleep_time': 0.001, 'use_tflite': True, 'use_pyarmnn': True,
                        'niter': 4, 'print_bool': True, 'save_dir': str(get_database('benchmarks', 'tmp'))})
    #execute_kwargs = {"report_dir": get_database('benchmarks','tmp'), 'device': device, 'sleep_time': 0.001})
    meas = result.sum()['time(ms)']
