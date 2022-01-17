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


class Graph_matcher():
    """Graph matcher"""
    
    def __init__(self, network, config, match=None):
        #make generator object
        self.network = network
        self.gen = generator.Graph_generator(network)
        self.gen.add_configfile(config)

        self.match = match
        self.df = {}
        self.column_names = ['width', 'height', 'channels', 'filters',
            'k_height', 'k_width', 'k_stride', 'pool_h', 'pool_w',
            'pool_stride', 'pad', 'batch_size', 'time(ms)', 'power_mean(mw)', 'f_pool',
            'f_bias', 'f_add', 'f_add_1', 'f_concat', 'f_act', 'f_pointwise']
        #make layer list
        print(self.gen.graph.model_spec['layers'])

    def run_bench(self, optimize = None, execute = None, parse = None, store = 10, hardware='ncs2'):
        config_len = len(self.gen.config)
        for i in range(config_len):
            print('-'*60)
            print('Running Config %i of %i' % (i, config_len))
            print('-'*60)
            self.gen.generate_graph_from_config(i)

            test_net = get_database('graphs','tf', self.network+'.pb')
            optimize(test_net, source_fw = "tf", network = self.network, image = None , save_folder = get_database('benchmarks','tmp'))

            #test_net = get_database('benchmarks','tmp', self.network+'.xml')
            #execute(test_net, report_dir = get_database('benchmarks','tmp'))

            #test_report = get_database('benchmarks','tmp','benchmark_average_counters_report.csv')
            test_report = get_database('benchmarks','tmp','timeline_01.json')
            report = parse(test_report) 

            self.match_and_add(self.gen.graph, report)
            if i % store == 0 and i > store-1 or i == config_len-1:
                print(i)
                for key, v in self.df_out.items():
                    try:
                        os.makedirs(get_database('benchmarks',hardware,'measurements'))
                    except:
                        pass
                    v.to_pickle(get_database('benchmarks',hardware,'measurements',key+'.p'))


    def match_and_add(self, graph, report):
        #compares graph with report
        logging.debug(report)
        logging.debug(graph.model_spec)
        missing_layers = []
        for l_name, l_attr in self.gen.graph.model_spec['layers'].items():
            # check if layer was executed
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

        #TODO make this a separate function?
        for l_name, l_attr in self.gen.graph.model_spec['layers'].items():
            if 'report_name' in l_attr:
                report_name = l_attr['report_name']
                if l_attr['type'] not in self.df.keys():
                    self.df[l_attr['type']] = []
                if l_name not in self.df.keys():
                    self.df[l_name] = []
                tmp = {}
                tmp['batch_size'] = l_attr['output_shape'][0]
                try:
                    tmp['height'] = l_attr['input_shape'][1]
                except:
                    pass
                try:
                    tmp['width'] = l_attr['input_shape'][2]
                except:
                    pass
                try:
                    tmp['channels'] = l_attr['input_shape'][3]
                except:
                    pass
                if l_attr['type'] == 'DataInput':
                    tmp.update({
                        'height': l_attr['output_shape'][1],
                        'width': l_attr['output_shape'][2],
                        'channels': l_attr['output_shape'][3],
                    })
                elif l_attr['type'] == 'Conv':
                    tmp.update({
                        'filters': l_attr['output_shape'][3],
                        'k_height': l_attr['kernel_shape'][0],
                        'k_width': l_attr['kernel_shape'][1],
                        'k_stride': l_attr['strides'][1],
                    })
                elif l_attr['type'] == 'DepthwiseConv':
                    tmp.update({
                        'filters': l_attr['output_shape'][3],
                        'k_height': l_attr['kernel_shape'][0],
                        'k_width': l_attr['kernel_shape'][1],
                        'k_stride': l_attr['strides'][1],
                    })
                elif l_attr['type'] == 'MatMul':
                    tmp.update({
                        'filters': l_attr['output_shape'][1],
                    })
                elif l_attr['type'] == 'Concat':
                    tmp.update({
                        'height': l_attr['output_shape'][1],
                        'width': l_attr['output_shape'][2],
                        'channels': l_attr['output_shape'][3],
                    })
                def add_to_tmp(report,report_name,key):
                    return report[key][report['name']==report_name].to_numpy()[0]

                tmp['time(ms)'] = add_to_tmp(report,report_name,'time(ms)')
                report = report[report['name']!=report_name]

                logging.debug(self.match)
                if l_name in self.match.keys():
                    for l_key in self.match[l_name]:
                        if l_key in missing_layers:
                            tmp[self.match[l_name][l_key]] = True
                fill = [tmp.get(x) for x in self.column_names]
                self.df[l_attr['type']].append(fill)
                self.df[l_name].append(fill)
                #Remove Layer for dataframe

            
        self.df_out = {}

        for l_type, l_df in self.df.items():
            self.df_out[l_type] = pd.DataFrame(self.df[l_type], columns=self.column_names)
            logging.debug(l_type)
            logging.debug(self.df_out[l_type])
        
        if len(report) > 0:
            logging.debug("Left Layers: %s" % report)
        else:
            logging.debug("No left layers")


    

    def convert_ncs2(self, report):
        logging.debug(report)
