# %%
from pathlib import Path
from annette.generation.layergen import LayerModelGen, HardwareModelGen
from annette.utils import get_database

import numpy as np
import pandas as pd
import pickle as pkl
import os

# %%
conv_est_dict = {'0': 'num_ops',
    '1': 'num_inputs',
    '2': 'num_outputs',
    '3': 'height',
    '4': 'width',
    '5': 'channels',
    '6': 'filters',
    '7': 'k_height',
    '8': 'k_width',
    '9': 'k_stride',
    '10': 'num_weights'}

tf_int = HardwareModelGen("ov_cpu_new")
tf_int.add_layer("conv", "Conv", "statistical", data = get_database('benchmarks', 'ov_cpu', 'annette_bench5', 'conv2d_1_Conv2D.p'),
    #sweep_data = get_database('benchmarks', 'ov_cpu', 'annette_bench0', 'Conv.p'),
    est_dict = conv_est_dict)
#tf_int.add_layer("conv2d", "Conv", "mixed", data = get_database('benchmarks', 'tf_basic_tmp', 'Conv.p'), est_dict = None)
#print(tf_int.layer_dict['conv2d'].data.head())

# %%
#tf_int.add_layer("conv2d", "Conv", "roofline", data = get_database('benchmarks', 'ov_cpu', 'annette_bench5', 'conv2d_1_Conv2D.p'),
#    #sweep_data = get_database('benchmarks', 'ov_cpu', 'annette_bench0', 'Conv.p'),
#    est_dict = conv_est_dict)
tf_int.add_layer("ncs2", "Conv", "roofline", data = get_database('benchmarks', 'ncs2', 'conv2d.p'),
    #sweep_data = get_database('benchmarks', 'ov_cpu', 'annette_bench0', 'Conv.p'),
    est_dict = conv_est_dict)

# %%
est_dict = {'0': 'num_ops',
    '1': 'num_inputs',
    '2': 'num_outputs',
    '3': 'height',
    '4': 'width',
    '5': 'channels',
    '6': 'filters',
    '7': 'k_height',
    '8': 'k_width',
    '9': 'k_stride',
    '10': 'num_weights'}



# %%
base = LayerModelGen("FullyConnected")
base.name = "FullyConnected"
base.layer_type = "FullyConnected"
base.estimation = "statistical"
base.read_data("database/benchmarks/ncs2/fully_connected.p")
base.data

# %%
est_dict = {'0': 'num_ops',
    '1': 'num_inputs',
    '2': 'num_outputs',
    '3': 'channels',
    '4': 'filters',
    '5': 'num_weights'}

print(base.generate_estimator(est_dict = est_dict, y_val='ops/s'))

base.gen_dict()
base.trans_Conv()
base.store_model("./test_data/generated/ncs2/fully_connected.sav")
base.to_json()

# %%
base = LayerModelGen("DepthwiseConv")
base.name = "DepthwiseConv"
base.layer_type = "DepthwiseConv"
base.estimation = "mixed"
base.read_data("database/benchmarks/ncs2/depthwise.p")
base.data

# %%
est_dict = {'0': 'num_ops',
    '1': 'num_inputs',
    '2': 'num_outputs',
    '3': 'height',
    '4': 'width',
    '5': 'channels',
    '6': 'k_height',
    '7': 'k_width',
    #'8': 'k_stride',
    '8': 'num_weights'}

print(base.generate_estimator(est_dict = est_dict, y_val='ops/s'))

base.gen_dict()
base.trans_Conv()
base.store_model("./test_data/generated/ncs2/depthwiseconv_stride.sav")
base.to_json()


