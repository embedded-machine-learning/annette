#%%
import pandas as pd
from annette.utils import get_database, write_result
import sys
sys.path.append("./")
from pathlib import Path
import logging
import os

import annette.graph as graph
from argparse import ArgumentParser
from annette.estimation.layer_model import Layer_model
from annette.estimation.mapping_model import Mapping_model


# define function to parse result.txt file
def parse_result_txt(file):
    # open file and read lines
    with open(file, 'r') as f:
        lines = f.readlines()

    # parse c_id table into pandas dataframe
    c_id_table_start = lines.index(' c_id  type                id       time (ms)\n')
    c_id_table_end = lines.index(' -------------------------------------------------\n')
    c_id_lines = lines[c_id_table_start+2:c_id_table_end]
    c_id_data = [line.strip().split() for line in c_id_lines]
    c_id_df = pd.DataFrame(c_id_data, columns=['c_id', 'type', 'id', 'time (ms)', 'percentage','%'])
    c_id_df['time (ms)'] = c_id_df['time (ms)'].astype(float)

    df = c_id_df.dropna()
    # sort by c_id
    df = df.sort_values(by=['c_id'])
    # reindex dataframe
    df = df.reset_index(drop=True)
    # drop % column and c_id column
    df = df.drop(columns=['%','c_id'])
    # conver time (ms) to float
    df['time (ms)'] = df['time (ms)'].astype(float)

    return df

# %%
ofa_folder = get_database('graphs','onnx','OFA_TEST1')
#get all subfolders
subfolders = [f for f in ofa_folder.iterdir() if f.is_dir()]
#get all subfolders of subfolders

subsubfolders = []
for n, name in enumerate(subfolders):
    print(n)
    # append all subfolders of subfolders
    subsubfolders.append([f for f in name.iterdir() if f.is_dir()])
    #subsubfolders = [f for f in subfolders[0].iterdir() if f.is_dir()]
# flatten list
subsubfolders = [item for sublist in subsubfolders for item in sublist]
print(subsubfolders)

# %%
#parse results for all subsubfolders
for subsubfolder in subsubfolders:
    print(subsubfolder)
    #check if exists
    if not Path(subsubfolder,'result.txt').exists():
        print('no result.txt')
        continue
    res = parse_result_txt(subsubfolder/'result.txt')
    #dump to csv
    res.to_csv(subsubfolder/'result.csv',index=False)
#res = parse_result_txt(get_database('graphs','onnx','OFA_TEST1','NUCLEO-L4R5ZI','ACSF1','result.txt'))
# %%
def test_ONNXGraph_to_annette(network="cf_resnet50",inputs=None):
    network_file = get_database('graphs','onnx',network+'.onnx')
    onnx_network = graph.ONNXGraph(network_file)
    annette_graph = onnx_network.onnx_to_annette(network, inputs)
    print(annette_graph)
    json_file = get_database( 'graphs', 'annette',
                     annette_graph.model_spec["name"]+'.json')
    json_file.parents[0].mkdir(parents=True, exist_ok=True)
    annette_graph.to_json(json_file)

    assert True
    return 0 

def run_estimate(model, name, hw_model = 'NUCLEO_big', mapping_model = 'simple'):
    '''
    Run model estimation for a specified model on several input points given as 
    config CSV file.
    '''

    NET_NAME = name

    RESULT_FILE_NAME = 'foobar.p'
    RESULT_FILE_NAME = None

    LAYER_MODEL = hw_model
    #MAPPING_MODEL = sys.argv[1]
    MAPPING_MODEL = mapping_model

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
    write_result(args.network, res, args.mapping,
                 args.layer, get_database('results'))

    return res


def main():
    print("main")
    for subsubfolder in subsubfolders:
        if not Path(subsubfolder,'result.txt').exists():
            print('no result.txt')
            continue
        else:
            #find network name based on onnx file
            onnx_file = [f for f in subsubfolder.iterdir() if f.suffix == '.onnx']
            # get filename without extension
            onnx_file = Path(onnx_file[0]).stem
            # get only only 3 childfolders

            subsubfolder = Path(*subsubfolder.parts[-3:])
            # check if subsubfolder contains L4R5ZI, L432KC or F746G
            if 'L4R5ZI' in str(subsubfolder):
                hw_model = 'NUCLEO_big'
            elif 'L432KC' in str(subsubfolder):
                hw_model = 'NUCLEO_small'
            elif 'F746G' in str(subsubfolder):
                hw_model = 'STM_DISCO'

            print(subsubfolder)
            #replace onnx with annette in subsubfolder
            annette_folder = str(subsubfolder).replace('onnx','annette')
            #create annette folder
            Path(annette_folder).mkdir(parents=True, exist_ok=True)
            net_name = str(subsubfolder)+'/'+onnx_file
            test_ONNXGraph_to_annette(str(subsubfolder)+'/'+onnx_file,['input'])
            est_graph = graph.AnnetteGraph(net_name,get_database('graphs','annette',net_name+'.json'))
            name = est_graph.model_spec["name"]

            estimate = run_estimate(est_graph, name, hw_model)
            print(name)
            print(estimate)

if __name__ == '__main__':
    main()
# %%
from annette.utils import get_database
from pathlib import Path
import pandas as pd
import numpy as np

def clean_and_group(res,est):
    res2 = res
    est2 = est
    # sort res2 by id
    res2 = res2.sort_values(by=['id'])
    # remove layers with type flatten and DataInput from est2
    est2 = est2[est2['type'] != 'Flatten']
    est2 = est2[est2['type'] != 'DataInput']
    # write into list of dataframes grouped by layer type
    pd.unique(res2['type']).tolist()
    pd.unique(est2['type']).tolist()

    types = {'DENSE': 'FullyConnected', 'POOL': 'Pool', 'CONV2D': 'Conv', 'CONCAT': 'Concat', 'NL': 'Relu', 'ELTWISE': 'Add'}
    resg = {}
    estg = {}
    for k, v in types.items():
        resg[v] = res2[res2['type'] == k].sort_values(by=['time (ms)'])
        estg[v] = est2[est2['type'] == v].sort_values(by=['time(ms)'])
    return resg, estg, types

ofa_folder = get_database('graphs','onnx','OFA_TEST1')
#get all subfolders
subfolders = [f for f in ofa_folder.iterdir() if f.is_dir()]
#get all subfolders of subfolders

subsubfolders = []
for n, name in enumerate(subfolders):
    print(n)
    # append all subfolders of subfolders
    subsubfolders.append([f for f in name.iterdir() if f.is_dir()])
    #subsubfolders = [f for f in subfolders[0].iterdir() if f.is_dir()]
# flatten list
subsubfolders = [item for sublist in subsubfolders for item in sublist]
print(subsubfolders)
res_dict = {'NUCLEO_big': {}, 'NUCLEO_small': {}, 'STM_DISCO': {}}
est_dict = {'NUCLEO_big': {}, 'NUCLEO_small': {}, 'STM_DISCO': {}}

# read results from csv and compare with annette results
for subsubfolder in subsubfolders:
    if not Path(subsubfolder,'result.txt').exists():
        print('no result.txt')
        continue
    else:
        #subsubfolder = Path(*subsubfolder.parts[-3:])

        #read csv
        res = pd.read_csv(subsubfolder/'result.csv')
        
        subsubfolder = Path(*subsubfolder.parts[-3:])
        if 'L4R5ZI' in str(subsubfolder):
            hw_model = 'NUCLEO_big'
        elif 'L432KC' in str(subsubfolder):
            hw_model = 'NUCLEO_small'
        elif 'F746G' in str(subsubfolder):
            hw_model = 'STM_DISCO'
        #print(subsubfolder)
        #read annette results
        result = get_database('results', hw_model, subsubfolder)
        #get csv file from result folder
        result = [f for f in result.iterdir() if f.suffix == '.csv']
        #read csv
        est = pd.read_csv(result[0])
        #print(est)
        
        #compare results
        print(res['time (ms)'].sum())
        print(est['time(ms)'].sum())
        print(subsubfolder.parts[-1])
        resg, estg, types = clean_and_group(res,est)
        res_dict[hw_model][subsubfolder.parts[-1]] = resg
        est_dict[hw_model][subsubfolder.parts[-1]] = estg



# %%

import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

# scatter plot of time for each layer type est vs res
for k, v in types.items():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=resg[v]['time (ms)'], y=estg[v]['time(ms)'], mode='markers', name='markers'))
    # add red line on 1st median
    max = np.max((resg[v]['time (ms)'].max(),estg[v]['time(ms)'].max()))
    fig.add_trace(go.Line(x=[0, max], y=[0, max], name='1:1 line'))
    fig.update_layout(title=f'{v} time(ms) est vs res', xaxis_title='res', yaxis_title='est')
    fig.show()


# %%

# put all results in one dataframe with layer type and est_time and res_time, dataset name and hw_model

df = pd.DataFrame(columns=['layer_type','est_time','res_time','dataset','hw_model'])
for hw_model, datasets in res_dict.items():
    for dataset, layers in datasets.items():
        for layer_type, layer in layers.items():
            df = df.append({'layer_type': layer_type, 'res_time': layer['time (ms)'].sum(), 'est_time': est_dict[hw_model][dataset][layer_type]['time(ms)'].sum(), 'dataset': dataset, 'hw_model': hw_model}, ignore_index=True)

# %%
print(df)
# %%
#plot scatterplott for estimated time vs res time for each layer type
df2 = df[df['hw_model'] == 'STM_DISCO']
fig = px.scatter(df2, x="res_time", y="est_time", color="layer_type", title='est vs res time for each layer type', hover_data=['layer_type','dataset'])
fig.show()
# %%
