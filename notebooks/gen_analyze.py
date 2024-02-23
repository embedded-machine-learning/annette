# %%
from pathlib import Path
from annette.generation.layergen import LayerModelGen, HardwareModelGen
from annette.utils import get_database

import numpy as np
import pandas as pd
import pickle as pkl
import os

# %%

hw_name = "rpi4"
hardware = "rpi4"
layer = "Conv"
benchmark = "annette_bench5"
layer_name = "conv"


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

tf_int = HardwareModelGen(hw_name)
tf_int.add_layer(layer_name, "Conv", "statistic", data = get_database('benchmarks', hardware, benchmark, layer+'.p'),
    #sweep_data = get_database('benchmarks', 'ov_cpu', 'annette_bench0', 'Conv.p'),
    est_dict = conv_est_dict)

# %%
data_analyze = tf_int.layer_dict[layer_name].data
data_analyze = data_analyze.astype(float)
# plot correlation matrix
import seaborn as sns
import matplotlib.pyplot as plt
corr = data_analyze.corr()
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns)
plt.show()

# feature importance estimation for ops/s
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import metrics

# perform feature selection
X = data_analyze.drop('ops/s', axis=1)
y = data_analyze['ops/s']
scaler = MinMaxScaler()
scaler_y = MinMaxScaler()
X = scaler.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1,1))
poly_X = PolynomialFeatures(2,interaction_only=True)
X = poly_X.fit_transform(X)
test_size = 0.1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)









# %%
list_columns = ['num_ops', 'num_inputs', 'num_outputs', 'height', 'width', 'channels', 'filters', 'k_height', 'k_width', 'k_stride', 'num_weights']
X = tf_int.layer_dict[layer_name].data[list_columns].values
y = tf_int.layer_dict[layer_name].data['ops/s'].values.reshape(-1,1)

# normalize data
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
#scaler = MinMaxScaler()
#scaler_y = MinMaxScaler()
#X = scaler.fit_transform(X)
#y = scaler_y.fit_transform(y)

#poly_X = PolynomialFeatures(2,interaction_only=True)

#X = poly_X.fit_transform(X)

# %%
test_size = 0.1
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import metrics

regressor = RandomForestRegressor(min_samples_leaf=1, max_depth=None, n_estimators=50, random_state=False, verbose=False, criterion='squared_error')
lookup_regressor = GradientBoostingRegressor(min_samples_leaf=1, max_depth=None, n_estimators=50, random_state=False, verbose=False, criterion='squared_error')
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

op_s = np.max(y_train)

regressor.fit(X_train[:,:], y_train[:]).score(X_test, y_test) #training the algorithm
lookup_regressor.fit(X_train[:,:], y_train[:]).score(X_test, y_test) #training the algorithm
y_pred = regressor.predict(X_test)
y_pred_lookup = lookup_regressor.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)/1e9)
print('R2 Score:', metrics.r2_score(y_test, y_pred))
print('Mean Absolute Error Scaled:', metrics.mean_absolute_error(y_test, y_pred)/op_s)
print(f'Mean abs. percentage error: {metrics.mean_absolute_percentage_error(y_test, y_pred) :.2%}')
print()

#metrics for lookup
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_lookup)/1e9)
print('R2 Score:', metrics.r2_score(y_test, y_pred_lookup))
print('Mean Absolute Error Scaled:', metrics.mean_absolute_error(y_test, y_pred_lookup)/op_s)
print(f'Mean abs. percentage error: {metrics.mean_absolute_percentage_error(y_test, y_pred_lookup) :.2%}')
print()

# %%
#write X and y back to dataframe

df = pd.DataFrame(X_test, columns=list_columns)
df['ops/s'] = y_test
df['ops/s_pred'] = y_pred

print(df)

# %%
#visualize the prediction
import plotly.express as px
import plotly.graph_objects as go

fig = px.scatter(df, x='ops/s', y='ops/s_pred',
                         hover_data=['num_ops', 'num_inputs', 'num_outputs',
                                     'height', 'width', 'channels', 'filters',
                                     'k_height', 'k_width', 'k_stride', 'num_weights'])


# visualize the prediction for lookup
fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test, y=y_pred_lookup,
                    mode='markers',
                    name='Test Data'))
fig.add_trace(go.Scatter(x=y_test, y=y_test,
                    mode='lines',
                    name='Ideal'))
fig.update_layout(title='Random Forest Regression',
                        xaxis_title='Measured',
                        yaxis_title='Predicted')
fig.show()



# %%
X = tf_int.layer_dict[layer_name].data[['num_ops', 'num_inputs', 'num_outputs', 'height', 'width', 'channels', 'filters', 'k_height', 'k_width', 'k_stride', 'num_weights']].values
y = tf_int.layer_dict[layer_name].data['ops/s'].values.reshape(-1,1)

from sklearn.manifold import TSNE
import plotly.express as px
from umap import UMAP


#tsne = TSNE(n_components=2, random_state=0)
#projections = tsne.fit_transform(X)

umap = UMAP(n_components=2, random_state=0, n_neighbors=10, min_dist=0.1, metric='euclidean')
projections_umap = umap.fit_transform(X)

#fig = px.scatter(
#    projections, x=0, y=1,
#    color=y.reshape(-1)
#    #labels={'color': 'species'}
#    )
#fig.show()

fig2 = px.scatter(
    projections_umap, x=0, y=1,
    color=y.reshape(-1)
    #labels={'color': 'species'}
    )
fig2.show()

# %%
# %%

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss, mean_squared_error


all_models = {}
common_params = dict(
    learning_rate=0.05,
    n_estimators=200,
    max_depth=2,
    min_samples_leaf=9,
    min_samples_split=9,
)
gbr_ls = GradientBoostingRegressor(loss="squared_error", **common_params)
all_models["mse"] = gbr_ls.fit(X_train, y_train)
for alpha in [0.05, 0.5, 0.95]:
    gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha, **common_params)
    all_models["q %1.2f" % alpha] = gbr.fit(X_train, y_train)

import matplotlib.pyplot as plt


y_pred = all_models["mse"].predict(X_test)
y_lower = all_models["q 0.05"].predict(X_test)
y_upper = all_models["q 0.95"].predict(X_test)
y_med = all_models["q 0.50"].predict(X_test)

#plt.plot(X_test, y_test, "b.", markersize=10, label="Test observations")
px.scatter(x=y_test, y=y_med, title="Predicted median")
px.scatter(x=y_test, y=y_pred, title="Predicted mean")
px.scatter(x=y_test, y=y_upper)
px.scatter(x=y_test, y=y_lower)

plt.show()

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


