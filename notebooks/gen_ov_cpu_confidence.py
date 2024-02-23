# %%
from pathlib import Path
from annette.generation.layergen import LayerModelGen, HardwareModelGen
from annette.utils import get_database
from sklearn.ensemble import GradientBoostingRegressor

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

common_params = dict(
    learning_rate=0.05,
    n_estimators=50,
    max_depth=None,
    min_samples_leaf=1
)

regressors = {}
alphas = [x/100 for x in range(10, 100, 10)]
print(alphas)

# %%
#for alpha in [0.05, 0.5, 0.95]:
for alpha in alphas:
    #if alpha == 0.5:
    #    regressors[alpha] = GradientBoostingRegressor(loss="squared_error", **common_params)
    #else:
    regressors[alpha] = GradientBoostingRegressor(loss="quantile", alpha=alpha, **common_params)

rpi4 = {}

for k, r in regressors.items():
    rpi4[k] = HardwareModelGen("rpi4"+str(k))
    rpi4[k].add_layer("conv", "Conv", "statistical", data = get_database('benchmarks', 'rpi4', 'annette_bench5', 'Conv.p'),
        #sweep_data = get_database('benchmarks', 'ov_cpu', 'annette_bench0', 'Conv.p'),
        regressor = r,
        est_dict = conv_est_dict)
#tf_int.add_layer("conv2d", "Conv", "mixed", data = get_database('benchmarks', 'tf_basic_tmp', 'Conv.p'), est_dict = None)
#print(tf_int.layer_dict['conv2d'].data.head())

# %%
import pandas as pd

data = {
    'Quant': [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9],
    'Metrics': ['ops/s', 'seconds', 'ops/s', 'seconds', 'ops/s', 'seconds', 'ops/s', 'seconds', 'ops/s', 'seconds', 'ops/s', 'seconds', 'ops/s', 'seconds', 'ops/s', 'seconds', 'ops/s', 'seconds'],
    'Mean Absolute Error': [2.2370869466990317, 0.008495684251184427, 1.273050702085591, 0.0036888793888724667, 0.9150399441109518, 0.0034527765941057185, 0.6901766635757892, 0.0020534466386083706, 0.45698206446307027, 0.0019356710926567706, 0.6300587291525312, 0.0023096160216120347, 0.7359536949948006, 0.002609452179295748, 1.1954718829663704, 0.004476572977551698, 2.1550842364955245, 0.006199655678564205],
    'Mean abs. percentage error': [20.95, 32.62, 12.56, 15.56, 11.61, 10.49, 10.48, 8.21, 17.55, 9.61, 27.56, 12.08, 33.93, 15.34, 47.94, 21.08, 79.56, 21.89]
}

df = pd.DataFrame(data)

print(df)

# %%

# run annette estimates for rpi4


import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

# make histogram sytle plot of the quantiles
g = sns.catplot(
    data=df, kind="bar",
    x="Quant", y="Mean abs. percentage error", hue="Metrics",
    ci="sd", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("Quantile", "Mean abs. percentage error")
g.legend.set_title("")
plt.title("Mean abs. percentage error for RPI4")
plt.savefig("rpi4_mae.png")

# %%
import json

network = "mobilenet_v1"
optimizer = "simple"
hws = [f"rpi4{alpha}" for alpha in alphas]
results = {}
df = pd.DataFrame()
# read results
for hw in hws:
    file = get_database('results', hw, f'{network}_{optimizer}.json')
    # read json file
    with file.open() as f:
        results[hw] = json.load(f)
    print(f"Read {hw} results")
    print(results[hw])
    print(results[hw]['sum'])






# %%
