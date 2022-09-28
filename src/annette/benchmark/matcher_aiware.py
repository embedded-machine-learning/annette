from __future__ import print_function
import json
from subprocess import CalledProcessError, TimeoutExpired
from threading import Lock
from time import ctime
import pandas as pd
import logging
from pathlib import Path
import os
from datetime import datetime as dt
import torch

from annette import get_database 
from annette.graph import AnnetteGraph
import annette.benchmark.generator as generator
from annette.hw_modules.hw_modules.aiware import optimize_network, run_network, r2a
from annette.hw_modules.hw_modules.aiware import get_layername_mapping, get_report_path
from annette.hw_modules.hw_modules.aiware import remove_tmp_files


CURR_TIME_STR = dt.now().strftime('%Y-%m-%d_%H-%M')  # Formatted start time of benchmark run

aiware_err_collection = []
aiware_err_collection_lock = Lock()

class Graph_matcher():
    """Graph matcher"""
    
    def __init__(self, network, config, net_dir='', match=None, matcher_idx=None):
        self.network = network
        self.gen = generator.Graph_generator(network, net_dir=net_dir, pad_pooling=False)
        self.gen.add_configfile(config)
        self.gen.export_pt_file = False
        self.keep_tmp_data = False

        self.matcher_idx = matcher_idx # "ID" for eacht concurrent matcher

        self.match = match
        self.df = {}
        self.column_names = ['conf_idx', 'width', 'height', 'channels', 'filters',
            'k_height', 'k_width', 'k_stride', 'k_dilation', 'pool_h', 'pool_w',
            'pool_stride', 'pad', 'batch_size', 'time(ms)', 'power_mean(mw)', 'f_pool',
            'f_bias', 'f_add', 'f_add_1', 'f_concat', 'f_act', 'f_pointwise', 'type']
        #make layer list
        print(self.gen.graph.model_spec['layers'])

    def run_bench(self, conf_indices=range(5), outlist=None, outlist_lock=None, errlist=None, errlist_lock=None):
        print('Running aiWare graph matcher benchmark...')
        for i, idx in enumerate(conf_indices):
            print(f'Running benchmark config: {idx} ...')
            _, pt_graph = self.gen.generate_graph_from_config(idx, framework='pytorch')
            pt_graph.cpu().eval()

            model_name = f'{CURR_TIME_STR}__{self.network}'
            inp_size = self.gen.get_torch_input_shape()
            name_idx = idx

            try:
                optimize_network(pt_graph, inp_size, model_name, name_idx=name_idx)
            except Exception as e:
                logging.error('Unexpected error during optimize_network():')
                logging.error(e)
                continue

            try:
                run_network(inp_size, model_name, name_idx=name_idx)
            except CalledProcessError as err: # aiware-estimator failed
                logging.error(f'ERROR while running network on config index {idx} on aiware-estimator:')
                logging.error(err.stdout)

                # Gather all aiware-estimator errors in list, s.t.
                # they can be printed at the end of the benchmak run:
                with aiware_err_collection_lock:
                    aiware_err_collection.append(f'{ctime()}\nConfig index: {idx}\n{err.stdout}')
                # Do the same for multiprocess version:
                if errlist is not None:
                    with errlist_lock:
                        errlist.append(f'{ctime()}\nConfig index: {idx}\n{err.stdout}')
                continue
            except TimeoutExpired:
                logging.error('Timeout error on aiware-estimator subprocess!')
                continue
            except Exception as e:
                logging.error('Unexpected error occured in function run_network():')
                logging.error(e)
                continue

            layername_map = get_layername_mapping(self.gen.graph)
            logging.debug(layername_map)

            report_path = os.path.join(
                get_report_path(model_name, name_idx),
                'layer_stats_0.csv'
            )
            try:
                report = r2a(report_path, layername_map)
                self.match_and_add(self.gen.graph, report, conf_idx=idx)
            except FileNotFoundError as e:
                logging.error(f'ERROR during parsing of result file for index {i}')
                logging.error(f'{e.filename} not found!')
            except Exception as e:
                logging.error('Unexpected error during r2a():')
                logging.error(e)

            if not self.keep_tmp_data:
                remove_tmp_files(model_name, name_idx)  # Cleanup files

            # Print short status message:
            print(f'Thread {self.matcher_idx}: Finished {(i+1) / len(conf_indices) :.2%}')

        if outlist is not None:
            with outlist_lock:
                outlist.append(self)

    def match_and_add(self, graph, report, conf_idx=None):
        # Compares graph with report
        logging.debug(report)
        logging.debug(graph.model_spec)
        missing_layers = []
        for l_name, l_attr in self.gen.graph.model_spec['layers'].items():
            # Check if layer was executed
            report_name = None
            if l_name in report['name'].to_numpy():
                report_name = l_name
                self.gen.graph.model_spec['layers'][l_name]['report_name'] = report_name
                #logging.debug("layer %s found " % l_name)
            elif len([l for l in report['name'].to_numpy() if l_name in l]) > 0:
                report_name = [l for l in report['name'].to_numpy() if l_name in l]
                report_name = report_name[0] 
                self.gen.graph.model_spec['layers'][l_name]['report_name'] = report_name
                #logging.debug("layer %s found " % l_name)
            elif l_name == 'Placeholder':
                if '<Extra>' in report['name'].to_numpy():
                    report_name = '<Extra>'
                    self.gen.graph.model_spec['layers'][l_name]['report_name'] = report_name
            else:
                missing_layers.append(l_name)
                #logging.debug("layer %s not found " % l_name)
        logging.debug("Missing layers %s" %missing_layers)

        for l_name, l_attr in self.gen.graph.model_spec['layers'].items():
            if 'report_name' in l_attr:
                report_name = l_attr['report_name']
                # if l_attr['type'] not in self.df.keys():
                #     self.df[l_attr['type']] = []
                if l_name not in self.df.keys():
                    self.df[l_name] = []
                tmp = {}
                tmp['batch_size'] = l_attr['output_shape'][0]
                try:
                    tmp['width'] = l_attr['input_shape'][1]
                except:
                    pass
                try:
                    if len(l_attr['input_shape']) == 3: # 1D operation
                        tmp_h = 1
                    else:
                        tmp_h = l_attr['input_shape'][2]
                    tmp['height'] = tmp_h
                except:
                    pass
                try:
                    if len(l_attr['input_shape']) == 3: # 1D operation
                        tmp_c = l_attr['input_shape'][2]
                    else:
                        tmp_c = l_attr['input_shape'][3]
                    tmp['channels'] = tmp_c
                except:
                    pass
                if l_attr['type'] == 'DataInput':
                    if len(l_attr['output_shape']) == 3:
                        tmp.update({
                            'width': l_attr['output_shape'][1],
                            'height': 1,
                            'channels': l_attr['output_shape'][2],
                        })
                    else:
                        tmp.update({
                            'width': l_attr['output_shape'][1],
                            'height': l_attr['output_shape'][2],
                            'channels': l_attr['output_shape'][3],
                        })
                elif l_attr['type'] in ['Conv', 'DepthwiseConv', 'ConvTranspose2d']:
                    tmp.update({
                        'filters': l_attr['output_shape'][3],
                        'k_height': l_attr['kernel_shape'][1],
                        'k_width': l_attr['kernel_shape'][0],
                        'k_stride': l_attr['strides'][1],
                        'k_dilation': l_attr['dilations'][1]
                    })
                elif l_attr['type'] in ['Conv1d', 'DepthwiseConv1d', 'ConvTranspose1d']:
                    tmp.update({
                        'filters': l_attr['output_shape'][2],
                        'k_height': 1,
                        'k_width': l_attr['kernel_shape'][0],
                        'k_stride': l_attr['strides'][1],
                        'k_dilation': l_attr['dilations'][1]
                    })
                elif l_attr['type'] == 'Pool':
                    tmp.update({
                        'pool_h': l_attr['kernel_shape'][1],
                        'pool_w': l_attr['kernel_shape'][2],
                        'pool_stride': l_attr['strides'][1],
                    })
                elif l_attr['type'] == 'Pool1d':
                    tmp.update({
                        'pool_h': 1,
                        'pool_w': l_attr['kernel_shape'][1],
                        'pool_stride': l_attr['strides'][1],
                    })
                elif l_attr['type'] == 'MatMul':
                    tmp.update({
                        'channels': l_attr['input_shape'][1],
                        'filters': l_attr['output_shape'][1],
                    })
                elif l_attr['type'] == 'Concat':
                    if len(l_attr['output_shape']) == 3: # 1D
                        tmp.update({
                            'width': l_attr['output_shape'][1],
                            'height': 1,
                            'channels': l_attr['output_shape'][2],
                        })
                    else:
                        tmp.update({
                            'width': l_attr['output_shape'][1],
                            'height': l_attr['output_shape'][2],
                            'channels': l_attr['output_shape'][3],
                        })

                def add_to_tmp(report, report_name, key):
                    return report[key][report['name']==report_name].to_numpy()[0]

                tmp['conf_idx'] = conf_idx  # Add used index of config
                tmp['type'] = l_attr['type']  # Add type of layers

                # Add all raw benchmark data to report:
                for col in list(report.columns):
                    if col not in self.column_names:
                        self.column_names.append(col)
                    if col not in tmp.keys():
                        tmp[col] = add_to_tmp(report, report_name, col)
                report = report[report['name'] != report_name]

                logging.debug(self.match)
                if self.match and l_name in self.match.keys():
                    for l_key in self.match[l_name]:
                        if l_key in missing_layers:
                            tmp[self.match[l_name][l_key]] = True

                fill = [tmp.get(x) for x in self.column_names]
                if l_attr['type'] in self.df.keys():
                    self.df[l_attr['type']].append(fill)
                self.df[l_name].append(fill)
            
        self.df_out = {}

        for l_type, l_df in self.df.items():
            self.df_out[l_type] = pd.DataFrame(l_df, columns=self.column_names)
            logging.debug(l_type)
            logging.debug(self.df_out[l_type])
        
        if len(report) > 0:
            logging.debug(f'Remaining layers: {report}')
        else:
            logging.debug('No remaining layers')
