# %%
from annette.graph import AnnetteGraph
from annette.estimation import layers 
from annette.estimation.roofline import Roofline_model
from annette.estimation.mixed import Mixed_model 
from annette.estimation.mapping import Mapping_model 
from pathlib import Path
import annette.utils as utils
from annette.generation.layergen import BaseModelGen 

# %%
import numpy as np
import pandas as pd
import pickle as pkl
import os

# %%
folder_in = 'database/benchmarks/dnndk/'
data = pd.read_pickle(os.path.join(folder_in,'conv2d_square.p'))
data1 = pd.read_pickle(os.path.join(folder_in,'conv2d_height.p'))
data2 = pd.read_pickle(os.path.join(folder_in,'conv2d_width.p'))
data = data.append(data1, ignore_index = True) 
data = data.append(data2, ignore_index = True) 

#data.to_pickle(os.path.join(folder_in,'conv2d.p'))

#folder_in = 'database/benchmarks/ncs2/'
#sweep_data = pd.read_pickle(os.path.join(folder_in,'conv2d_sweep.p'))

# %%
"""
{'c_par': 16.0,
 'c_alpha': 1.0048811612723485,
 'f_par': 16.0,
 'f_alpha': 0.8813084630717796,
 'w_par': 2.0,
 'w_alpha': -0.17930082420954604,
 'h_par': 8.0,
 'h_alpha': 0.48589642466149335}
"""
base = BaseModelGen("base")

# %%
base.read_data("database/benchmarks/dnndk/conv2d_square.p")
print(len(base.data))
base.read_data("database/benchmarks/dnndk/conv2d_height.p")
print(len(base.data))
base.read_data("database/benchmarks/dnndk/conv2d_width.p")
print(len(base.data))
base.read_data("database/benchmarks/dnndk/conv2d_stride2.p")
print(len(base.data))

# %%
base.data

# %%
data = base.data
data['num_ops'] = data['k_height']*data['k_width']*data['height']*data['width']*data['channels']*data['filters']*2/data['k_stride']/data['k_stride']
data['num_inputs'] = data['height']*data['width']*data['channels']
data['num_outputs'] = data['height']*data['width']*data['filters']/data['k_stride']/data['k_stride']
data['num_weights'] = data['k_height']*data['k_width']*data['filters']*data['channels']
data['ops/s'] = data['num_ops']/(data['time(ms)']/1e3)

# %%
data
data.to_pickle(os.path.join(folder_in,'conv2d_all.p'))

# %%
base.data = data

# %%
data['ops/s'].max()

# %%
#import plotly.express as px


#fig = px.scatter(data, x="num_ops", y="ops/s",hover_name="channels",
#                 hover_data=["channels","filters","height","width"],
#                 color="channels"
#                )
#fig.show()

# %%
base.max_bandwidth()
base.data

# %%
print(base.generate_roofline())
base.gen_dict()
base.to_json()

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
    '9': 'num_weights',
    '10': 'k_stride'}

print(base.generate_estimator(est_dict = est_dict, y_val='ops/s'))
base.name = "conv2d"
base.type = "Conv"
base.estimation = "statistical"
base.gen_dict()
base.trans_Conv()
base.layer_dict
base.store_model("./database/dnndk/conv2d_stride.sav")

# %%
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

est_dict = {'0': 'num_ops',
    '1': 'num_inputs',
    '2': 'num_outputs',
    '3': 'height',
    '4': 'width',
    '5': 'channels',
    '6': 'filters',
    '7': 'k_height',
    '8': 'k_width',
    '9': 'num_weights'}

#regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=None, criterion='mse'),n_estimators=50, random_state=False)

#print(base.generate_estimator(est_dict = est_dict, regressor = regr_2))

from sklearn import linear_model
regr_2 = linear_model.Lasso(alpha=0.1)

#print(base.generate_estimator(est_dict = est_dict, regressor = regr_2, y_val='time(ms)'))
print(base.generate_estimator(est_dict = est_dict, regressor = regr_2, y_val='time(ms)',poly=3))

# %%
base.store_model("./database/ncs2/conv2_poly3.sav")
base.to_json()

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
    '9': 'num_weights'}

print(base.generate_estimator(est_dict = est_dict))

# %%
base.store_model("./database/ncs2/conv2d.sav")

# %%

in_layer = BaseModelGen("base")

in_layer.read_data("database/benchmarks/ncs2/input.p")
print(len(in_layer.data))
data = in_layer.data
data['num_inputs'] = data['height']*data['width']*data['channels']
data['num_outputs'] = data['height']*data['width']*data['channels']
data['num_weights'] = 0
data['num_ops'] = 0
data['out/s'] = data['num_outputs']/(data['time(ms)']/1e3)
in_layer.data

# %%
est_dict = {
    '0': 'num_inputs',
    '1': 'height',
    '2': 'width',
    '3': 'channels'}

print(in_layer.generate_estimator(est_dict = est_dict, y_val = "out/s"))

# %%
in_layer.gen_dict()
print(in_layer.to_json())

# %%
from annette.graph import AnnetteGraph
from annette.estimation import layers 
from annette.estimation.roofline import Roofline_model
from annette.estimation.mixed import Mixed_model 
from annette.estimation.mapping import Mapping_model 
from pathlib import Path
import annette.utils as utils
from annette.generation.layergen import BaseModelGen 

import numpy as np
import pandas as pd
import pickle as pkl
import os

# %%
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier

# %%
folder_in = 'database/benchmarks/dnndk/'
data = pd.read_pickle(os.path.join(folder_in,'conv2d_sweep.p'))

# %%
base = BaseModelGen("base")
base.read_data("database/benchmarks/dnndk/conv2d_sweep.p")
print(len(base.data))
print(base.data)
data.columns

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
    '9': 'num_weights'}

print(base.generate_estimator(est_dict = est_dict))

# %%
import plotly.express as px

channels = 1024
f = ((data['channels'] <= channels) & (data['channels'] > 0) & (data['k_height'] == 3)  
        & (data['height'] == 32)
        & (data['width'] == 32)
        #(data['channels'] < 24) &
        #(data['filters'] < 24)
    )
f_data = data[f]

fig = px.scatter(f_data, x="channels", y="time(ms)",hover_name="channels",
                 hover_data=["channels","filters","inputs","height","width"],
                 color="filters",
                 #animation_frame="filters",
                 color_continuous_scale=px.colors.diverging.Tealrose,
                 title = "channels from 1 to "+str(channels)+", height = 32, width = 32")
fig.show()

# %%

f_data2 = f_data

# %%
num = 16
f_data['ueff'] = ((f_data['channels']/num)/np.ceil(f_data['channels']/num))
alpha = 0.75
f_data['ueff_alpha'] = (1-alpha) + f_data['ueff']*alpha
f_data2['time16'] = f_data['time(ms)']*f_data['ueff_alpha']

# %%

fig = px.scatter(f_data2, x="channels", y="ueff",hover_name="channels",
                 hover_data=["channels","filters","inputs","height","width"],
                 color="filters",
                 #animation_frame="filters",
                 color_continuous_scale=px.colors.diverging.Tealrose,
                 title = "channels from 1 to "+str(channels)+", height = 32, width = 32")
fig.show()

# %%

fig = px.scatter(f_data2, x="channels", y="ops/s",hover_name="channels",
                 hover_data=["channels","filters","inputs","height","width"],
                 color="filters",
                 #animation_frame="filters",
                 color_continuous_scale=px.colors.diverging.Tealrose,
                 title = "channels from 1 to "+str(channels)+", height = 32, width = 32")
fig.show()

# %%

fig = px.scatter(f_data2, x="channels", y="time16",hover_name="channels",
                 hover_data=["channels","filters","inputs","height","width"],
                 color="filters",
                 #animation_frame="filters",
                 color_continuous_scale=px.colors.diverging.Tealrose,
                 title = "channels from 1 to "+str(channels)+", height = 32, width = 32")
fig.show()


