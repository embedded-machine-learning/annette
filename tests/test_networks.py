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
from annette.estimation.layer_model import Layer_model
from annette.estimation.mapping_model import Mapping_model



#logging.basicConfig(level=logging.DEBUG)

def run_estimate(model, layer=None, mapping=None, result_file=None):
    '''
    Run model estimation for a specified model on several input points given as 
    config CSV file.
    '''
    NET_NAME = 'cf_reid'

    RESULT_FILE_NAME = 'foobar.p'
    RESULT_FILE_NAME = None

    if layer is None:
        LAYER_MODEL = 'ov_cpu_roofline'
    else:
        LAYER_MODEL = layer
    
    if mapping is None:
        MAPPING_MODEL = 'simple'
    else:
        MAPPING_MODEL = mapping

    # Set internal args needed for estimate function:
    args = ArgumentParser()
    args.network = f'{NET_NAME}'
    args.layer = LAYER_MODEL
    args.mapping = MAPPING_MODEL
    if args.mapping != "none":
        opt = Mapping_model.from_json(
            get_database('models', 'mapping', args.mapping+'.json'))
        opt.run_optimization(model)

    # LOAD MODELS
    mod = Layer_model.from_json(
        get_database('models', 'layer', args.layer+'.json'))

    # APPLY ESTIMATION
    res = mod.estimate_model(model)

    return res

# %%

def compare_est_res(res, est, dataset):
    '''
    Compare estimation results with benchmark results.
    '''
    # write res and est to dataframe by matching names in res and est
    df = pd.DataFrame(columns=['dataset','name','meas', 'est', 'type', 'missing'])
    df_scaled = pd.DataFrame(columns=['dataset','name','meas', 'est', 'type', 'missing'])

    # replace all / with _ in name of res and est
    res['name'] = res['name'].str.replace('/', '_')
    est['name'] = est['name'].str.replace('/', '_')

    # iterate over res and est and append to if name matches
    # count if there was a match and if not, append to error list
    est_layers = est['name'].tolist()
    for i in range(len(res)):
        count = 0
        for j in range(len(est)):
            if res.iloc[i]['name']== est.iloc[j]['name']:
                df = pd.concat([df, pd.DataFrame({'dataset': dataset, 'name': res.iloc[i]['name'], 'type': est.iloc[j]['type'], 'meas': res.iloc[i]['time(ms)'], 'est': est.iloc[j]['time(ms)'], 'missing': 0}, index=[0])], ignore_index=True)
                count += 1
                est_layers.remove(est.iloc[j]['name'])

        if count == 0:
            print('No match for', res.iloc[i]['name'])
            df = pd.concat([df, pd.DataFrame({'dataset': dataset, 'name': res.iloc[i]['name'], 'type': 'missing', 'meas': res.iloc[i]['time(ms)'], 'est': 0, 'missing': 1}, index=[0])], ignore_index=True)
    for i in range(len(est_layers)):
        df = pd.concat([df, pd.DataFrame({'dataset': dataset, 'name': est_layers[i], 'type': est.iloc[i]['type'], 'meas': 0, 'est': est.iloc[i]['time(ms)'], 'missing' : 1}, index=[0])], ignore_index=True)

    df_missing = df.loc[df['missing'] == 1]

    #get missing estimates
    df_est = df_missing.loc[df['type'] != 'missing']
    df_meas = df_missing.loc[df['type'] == 'missing']
    for index, est in df_est.iterrows():
        #see if there is a measurement where the estimate name is a substring of the measurement name
        df_meas_tmp = df_meas.loc[df_meas['name'].str.contains(est['name'], na=False)]
        if len(df_meas_tmp) == 1:
            # match found and combine the two rows and add to df 
            df = df.query(f"name != '{est['name']}'")
            df = df.query(f"name != '{df_meas_tmp['name'].values[0]}'")
            est['missing'] = 0
            est['meas'] = df_meas_tmp['meas'].values[0]
            #est['name'] = df_meas_tmp['name']
            # append est to df
            l = est.to_frame()
            l = l.transpose()
            df = pd.concat([df, l], ignore_index=True)
            # remove meas from df_meas
            df_meas = df_meas.drop(df_meas_tmp.index)
    
    # normalize data
    df_scaled = df.copy()
    df_scaled['est'] = df_scaled['est'] / df_scaled['meas'].max()
    df_scaled['meas'] = df_scaled['meas'] / df_scaled['meas'].max()
    return df, df_scaled

#working INTEL_CPU
#  (squeezenet1_0)
def compute_error_metrics(name, meas, est):
    df = pd.DataFrame(columns=['name','meas', 'est', 'mae', 'mse', 'rmse', 'mape', 'mspe', 'rmspe'])
    df = pd.concat([df, pd.DataFrame({'name': name, 'meas': meas, 'est': est}, index=[0])], ignore_index=True)
    # compute error metrics and feed to dataframe
    # select row with name
    row = df.loc[df['name'] == name]
    # compute error metrics
    mae = np.abs(row['meas'] - row['est'])
    mse = (row['meas'] - row['est'])**2
    rmse = np.sqrt(mse)
    mape = np.abs((row['meas'] - row['est']) / row['meas']) * 100
    mspe = ((row['meas'] - row['est']) / row['meas'])**2 * 100
    rmspe = np.sqrt(mspe)
    # feed to dataframe
    df.loc[df['name'] == name, 'mae'] = mae
    df.loc[df['name'] == name, 'mse'] = mse
    df.loc[df['name'] == name, 'rmse'] = rmse
    df.loc[df['name'] == name, 'mape'] = mape
    df.loc[df['name'] == name, 'mspe'] = mspe
    df.loc[df['name'] == name, 'rmspe'] = rmspe
    return df

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def main():

    test = False
    visualize = True 

    networks=["cf_openpose", "cf_reid", "cf_landmark", "cf_inceptionv1", "cf_inceptionv2", "cf_inceptionv3", "cf_inceptionv4", "cf_squeezenet"]
    #networks = ["pt_resnet50", "cf_openpose", "cf_inceptionv3"]
    #networks = ["cf_inceptionv3"]
    config="dummy.csv"
    layer = 'ncs2-mixed'
    #layer = 'ov_cpu_mixed'

    mapping = 'ov2'
    #mapping = 'simple'

    device = 'MYRIAD'
    #device = 'CPU'


    if test is True:
        for network in networks:

            graph_matcher = matcher.Graph_matcher(network, config)
            result = graph_matcher.run_bench(optimize = ncs2.inference.optimize_network, execute = ncs2.inference.run_network_new, parse = ncs2.parser.r2a,
                execute_kwargs = {"report_dir": get_database('benchmarks','tmp'), 'device': device, 'sleep_time': 0.001})
            estimate = run_estimate(graph_matcher.gen.graph, layer, mapping)
            meas = result.sum()['time(ms)']
            est = estimate[2].sum()['time(ms)']
            print(result)
            print(estimate[2])
            print(meas, est)

            # create dataframe with error metrics and measurements and estimates
            tmp_errors = compute_error_metrics(network, meas, est)
            tmp_df, tmp_df_scaled = compare_est_res(result, estimate[2], network)

            # append to df
            if 'df' not in locals():
                df = tmp_df; df_scaled = tmp_df_scaled
            else:
                df = pd.concat([df, tmp_df], ignore_index=True)
                df_scaled = pd.concat([df_scaled, tmp_df_scaled], ignore_index=True)

            # append to errors
            if 'errors' not in locals():
                errors = tmp_errors
            else:
                errors = pd.concat([errors, tmp_errors], ignore_index=True)


            print(df, df_scaled, errors)

            # store dataframes
            df.to_csv(f'df_{mapping}_{layer}_{device}.csv')
            df_scaled.to_csv(f'df_scaled_{mapping}_{layer}_{device}.csv')
            errors.to_csv(f'errors_{mapping}_{layer}_{device}.csv')
    folder = '/home/mwess/tmp_tut_merge/SoC_EML_ANNETTE/'

    if visualize is True:
        # load dataframes
        df = pd.read_csv(f'{folder}df_{mapping}_{layer}_{device}.csv')
        df_scaled = pd.read_csv(f'{folder}df_scaled_{mapping}_{layer}_{device}.csv')
        errors = pd.read_csv(f'{folder}errors_{mapping}_{layer}_{device}.csv')
        # make subplots
        # add scatter plot
        fig = px.scatter(df_scaled, x="meas", y="est", color="type",facet_col="dataset", facet_col_wrap=4, title="Measured vs. Estimated Time", hover_data=['name'])

        # add diagonal line for all subplots
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='black', width=1)), row='all', col='all', exclude_empty_subplots=True)

        # add title and labels for all facets
        fig.update_layout(title_text="Measured vs. Estimated Time", xaxis_title="Measured Time (ms)", yaxis_title="Estimated Time (ms)")
        fig.show()
        fig.write_html(f"{folder}{mapping}_{layer}_{device}_scatter.html")

        fig2 = px.scatter(df, x="meas", y="est", color="type",facet_col="dataset", facet_col_wrap=4, title="Measured vs. Estimated Time", hover_data=['name'])
        fig2.update_layout(title_text="Measured vs. Estimated Time", xaxis_title="Measured Time (ms)", yaxis_title="Estimated Time (ms)")
        fig2.show()
        fig2.write_html(f"{folder}{mapping}_{layer}_{device}_unscaled.html")

        print(errors)
        # add bar plot
        fig = px.bar(errors, x="name", y=["mape", "mspe", "rmspe"], hover_data=["meas", "est"], title="Error Metrics")
        fig.show()
        fig.write_html(f"{folder}{mapping}_{layer}_{device}_bar.html")

    return


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)

    main()


# %% plot
# plot results against est in scatter plot
#import plotly.express as px
#import plotly.graph_objects as go
#fig = px.scatter(df, x="meas", y="est", hover_data=['name'], color="type")
#fig.show()

# %%
