import pickle
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from collections.abc import Mapping

from annette import get_database 

__author__ = "Matthias Wess"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"


def write_result(network, res_dict, model, hardware, folder='database'):
    """Write results to a file in a folder .

    Args:
        network ([type]): [description]
        res_dict ([type]): [description]
        model ([type]): [description]
        hardware ([type]): [description]
        folder (str, optional): [description]. Defaults to 'database'.
    """
    try:
        os.mkdir(Path(folder))
    except:
        pass
    try:
        os.mkdir(Path(folder,hardware))
    except:
        pass
    with open(Path(folder,hardware, network+'_'+model+'.json'), 'w+') as json_file:
        temp = {}
        temp["layers"] = res_dict[1]
        temp["sum"] = res_dict[0]
        temp["hardware"] = hardware
        temp["network"] = network
        temp["model"] = model
        json.dump(temp, json_file, indent=4)

    """
    with open(Path(folder, hardware, network+'_'+model+'_detail.json'), 'w+') as json_file:
        temp = res_dict[2].to_json()
        json.dump(temp, json_file, indent = 4)
    """

    print("Results stored in %s " %
          Path(folder, hardware, network+'_'+model+'.json'))


def read_result(network, model, hardware, folder='database'):
    """Reads the result of a network run

    Args:
        network ([type]): [description]
        model ([type]): [description]
        hardware ([type]): [description]
        folder (str, optional): [description]. Defaults to 'database'.

    Returns:
        [type]: [description]
    """
    data = [None] * 3

    with open(Path(folder, hardware, network+'_'+model+'.json')) as json_file:
        print(json_file)
        temp = json.load(json_file)
        # print(data)
        data[0] = temp['sum']
        data[1] = temp['layers']

    try:
        with open(Path(folder, hardware, network+'_'+model+'_detail.json'), 'r+') as json_file:
            temp = json.load(json_file)
            data[2] = pd.read_json(temp)
    except:
        print("No detail Data")
    return data


def bench_to_annette(in_dict):

    for k, v in in_dict.items():
        if isinstance(v, Mapping):
            v = bench_to_annette(v)
        else:
            if v == "height":
                in_dict[k] = {"name": "input_shape", "i": 1}
            if v == "width":
                in_dict[k] = {"name": "input_shape", "i": 2}
            if v == "channels":
                in_dict[k] = {"name": "input_shape", "i": 3}
            if v == "filters":
                in_dict[k] = {"name": "output_shape", "i": 3}
            if v == "k_height":
                in_dict[k] = {"name": "kernel_shape", "i": 0}
            if v == "k_width":
                in_dict[k] = {"name": "kernel_shape", "i": 1}
    return in_dict


def ncs2_to_format(df):
    """Format ncs2 results as to annette format

    Args:
        df ([type]): [description]

    Returns:
        [type]: [description]
    """
    # print(df)
    # filter only executed Layers
    f = (df['ExecStatus'] == 'EXECUTED')
    df = df[f]
    # print(df)
    df['LayerName'] = df['LayerName'].str.replace('/', '_', regex=True)
    df['LayerName'] = df['LayerName'].str.replace('-', '_', regex=True)

    ncs2 = df[['LayerName', 'LayerType', 'RunTime(ms)']]
    ncs2 = ncs2.rename(
        columns={'RunTime(ms)': 'measured', 'LayerName': 'name', 'LayerType': 'type'})
    return ncs2


def ncs2_file_to_format(file):
    """Load ncs2 file into a annette format

    Args:
        file ([type]): [description]

    Returns:
        [type]: [description]
    """
    df = pickle.load(open(file, "rb"))

    return ncs2_to_format(df)


def dnndk_to_format(inp):
    df = inp

    df['LayerName'] = df['LayerName'].map(str)
    df['LayerName'] = df['LayerName'].str.replace('/', '_', regex=True)

    dnndk = df[['LayerName', 'RunTime(ms)']]
    dnndk = dnndk.rename(
        columns={'RunTime(ms)': 'measured', 'LayerName': 'name'})

    #dnndk = pd.DataFrame({'name':[],'measured':[],'utilization':[],'MB/S':[],'Workload(MOP)':[]})
    return dnndk


def dnndk_file_to_format(file):
    inp = pickle.load(open(file, "rb"))
    print(inp)
    df = inp['layer']

    df['LayerName'] = df['LayerName'].str.replace('/', '_', regex=True)

    dnndk = df[['LayerName', 'RunTime(ms)']]
    dnndk = dnndk.rename(
        columns={'RunTime(ms)': 'measured', 'LayerName': 'name'})

    #dnndk = pd.DataFrame({'name':[],'measured':[],'utilization':[],'MB/S':[],'Workload(MOP)':[]})
    return dnndk


def add_estimation(result_df, est_df, name='predicted'):
    """Add an estimation to result_df .

    Args:
        result_df ([type]): [description]
        est_df ([type]): [description]
        name (str, optional): [description]. Defaults to 'predicted'.

    Returns:
        [type]: [description]
    """

    # filter only available Layers
    est_df = est_df.dropna(subset=['name', 'time(ms)'])
    est_df['name'] = est_df['name'].str.replace('/', '_', regex=True)
    est_df['name'] = est_df['name'].str.replace('-', '_', regex=True)

    if not('num_ops' in result_df.columns):
        result_df["num_ops"] = np.nan
        result_df["type"] = np.nan
        result_df["num_inputs"] = np.nan
        result_df["num_outputs"] = np.nan
        result_df["num_weights"] = np.nan
        result_df["num_features"] = np.nan
        result_df["num_data"] = np.nan
        result_df[name] = np.nan

    est_df2 = est_df

    print(result_df)
    for index, row in est_df.iterrows():
        # See if estimated Layer fits
        #print("Name: ",i[0])
        prev = None
        if row['type'] == "DataInput":
            prev = row['name']
            row['name'] = "<Extra>"
        f = (row['name'] == result_df['name'])
        # print(len(result_df[f]['measured']))
        if(len(result_df[f]['name'])):
            result_df.loc[f, name] = row['time(ms)']
            result_df.loc[f, 'num_ops'] = row['num_ops']
            result_df.loc[f, 'type'] = row['type']
            result_df.loc[f, 'num_inputs'] = row['num_inputs']
            result_df.loc[f, 'num_outputs'] = row['num_outputs']
            result_df.loc[f, 'num_weights'] = row['num_weights']
            result_df.loc[f, 'num_features'] = row['num_inputs'] + \
                row['num_outputs']
            result_df.loc[f, 'num_data'] = row['num_inputs'] + \
                row['num_outputs']+row['num_weights']
            est_df2 = est_df2[est_df2['name'] != row['name']]
            est_df2 = est_df2[est_df2['name'] != prev]
            # print("fits")
            # print(result_df[f]['measured'].values[0],i[1],"\n")
            #ncs2.loc[len(ncs2)] = {'LayerName':i[0], 'measured':i[4], 'predicted':est[f]['time(ms)'].values[0],'type':i[2]}
        else:
            print("not found")
            # print(row)
            if row['name'].endswith("_depthwise_depthwise"):
                print("depthwise detected")

            result_df = result_df.append(pd.Series(), ignore_index=True)
            result_df.loc[result_df.index[-1], 'name'] = row['name']
            result_df.loc[result_df.index[-1], 'type'] = row['type']
            result_df.loc[result_df.index[-1], 'num_ops'] = row['num_ops']
            result_df.loc[result_df.index[-1],
                          'num_inputs'] = row['num_inputs']
            result_df.loc[result_df.index[-1],
                          'num_outputs'] = row['num_outputs']
            result_df.loc[result_df.index[-1],
                          'num_weights'] = row['num_weights']
            result_df.loc[result_df.index[-1],
                          'num_features'] = row['num_inputs']+row['num_outputs']
            result_df.loc[result_df.index[-1], 'num_data'] = row['num_inputs'] + \
                row['num_outputs']+row['num_weights']
            result_df.loc[result_df.index[-1], name] = row['time(ms)']
            pass
            # print("doesnt")
            # print(i[1],"\n")

    print(result_df.tail(20))
    print("Estimated Rest", len(est_df2), "")
    return result_df
