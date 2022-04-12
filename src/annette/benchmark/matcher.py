from __future__ import print_function
from copy import deepcopy
import json
import numpy as np
import pandas as pd
import pickle as pkl
import logging
from pathlib import Path
import os
from scipy.signal import savgol_filter, medfilt, correlate, windows
from scipy.ndimage import maximum_filter
import multiprocessing

from annette import get_database 
from annette.graph import AnnetteGraph
import annette.benchmark.generator as generator
from powerutils import processing, measurement


import matplotlib.pyplot as plt


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
            'pool_stride', 'pad', 'batch_size', 'f_pool',
            'f_bias', 'f_add', 'f_add_1', 'f_concat', 'f_act', 'f_pointwise', 'time(ms)', 'mean(W)', 'mult(mJ)']
        #make layer list
        print(self.gen.graph.model_spec['layers'])

    def run_bench(self, optimize = None, execute = None, parse = None, store = 5, hardware='ncs2', start=0, vis=False):
        config_len = len(self.gen.config)
        assert(start >= config_len, "Selected starting number {s} larger than config length {l}".format(s=start,l=config_len))

        for i in range(start, config_len):
            print('-'*60)
            print('Running Config %i of %i' % (i, config_len))
            print('-'*60)
            self.gen.generate_graph_from_config(i)
            rate = 500
            vis = True

            test_net = get_database('graphs','tf', self.network+'.pb')
            optimize(test_net, source_fw = "tf", network = self.network, input_shape = None , save_folder = get_database('benchmarks','tmp'))

            test_net = get_database('benchmarks','tmp', self.network+'.xml')

            # gather power data into .dat file
            #pm = measurement.power_measurement(sampling_rate=rate*1000, data_dir="./tmp", max_duration=60)
            #pm_kwargs = {"model_name": "tmpmodel"}
            #pm.start_gather(pm_kwargs) # power data aquisation

            execute_kwargs = {"xml_path": test_net, "report_dir": get_database('benchmarks','tmp'), 'device': 'MYRIAD', 'sleep_time': 0.001}
            #execute(test_net, report_dir = get_database('benchmarks','tmp'))
            #dur, power_dir, power_file = execute(**execute_kwargs)
            p = multiprocessing.Process(target=execute,kwargs=execute_kwargs)
            p.start()
            p.join(60)
            if p.is_alive():
                print("Execution seem stuck!")
                p.terminate()
                p.join()
            elif p.exitcode != 0:
                print("Execution failed with exitcode: {}".format(p.exitcode))
            else:
                print("Execution run successfully")
                # end power measurement
                #pm.end_gather(True)  # stop the power measurement
                #store dat file
                #dat_file_path = pm.dat_filepath # filepath of the dat file with the measurement

                test_report = get_database('benchmarks','tmp','benchmark_average_counters_report.csv')
                #test_report = get_database('benchmarks','tmp','timeline_01.json')
                report = parse(test_report) 
                print(report)
                duration = np.sum(report['time(ms)'])
                #total_result = processing.extract_power_profile(pm.dat_filename, pm.data_dir, duration, sample_rate = rate)
                
                pad = int(np.min((np.sqrt(duration)*200,2000)))

                print("duration: ",duration)
                print("pad : ",pad)

                result = processing.unite_latency_power_meas(report, 'test_infmod.dat', 'tmp/', sample_rate = rate, padding=pad, vis=vis)

                self.match_and_add(self.gen.graph, result[0])
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
            logging.debug("L_name: %s" % l_name)
            #print([l for l in report['name'].to_numpy(dtype=str) if l.find(l_name) != 1])
            if l_name in report['name'].to_numpy(dtype=str):
                report_name = l_name
                self.gen.graph.model_spec['layers'][l_name]['report_name'] = report_name
                #logging.debug("layer %s found " % l_name)
            elif 'Placeholder' in l_name:
                print(report)
                if 'NaN' in report['name'].to_numpy():
                    report_name = 'NaN'
                    self.gen.graph.model_spec['layers'][l_name]['report_name'] = report_name
            elif len([l for l in report['name'].to_numpy(dtype=str) if l.find(l_name) != 1]) > 0:
                report_name = [l for l in report['name'].to_numpy(dtype=str) if l.find(l_name) != 1]
                report_name = report_name[0] 
                self.gen.graph.model_spec['layers'][l_name]['report_name'] = report_name
                logging.debug("layer %s found " % l_name)
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
                    return report[key][report['name']==report_name].to_numpy(dtype=str)

                tmp['time(ms)'] = add_to_tmp(report,report_name,'time(ms)')
                tmp['mult(mJ)'] = add_to_tmp(report,report_name,'mult(mJ)')
                tmp['mean(W)'] = add_to_tmp(report,report_name,'mean(W)')
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

def measure_annette_network(optimize, execute, parse, network):
        rate = 500
        vis = False

        gen = generator.Graph_generator(network)
        gen.generate_graph_from_config()
        test_net = get_database('graphs','tf', network+'.pb')

        optimize(test_net, source_fw = "tf", network = network, input_shape = None , save_folder = get_database('benchmarks','tmp'))

        test_net = get_database('benchmarks','tmp', network+'.xml')
        execute_kwargs = {"xml_path": test_net, "report_dir": get_database('benchmarks','tmp'), 'device': 'MYRIAD'}
        dur, power_dir, power_file = execute(**execute_kwargs)

        test_report = get_database('benchmarks','tmp','benchmark_average_counters_report.csv')
        report = parse(test_report) 
        duration = np.sum(report['time(ms)'])
        result = processing.unite_latency_power_meas(report, power_file+'.dat', power_dir, sample_rate = rate, vis=vis, padding=200)
        print(result)
        print(len(result[1]))

        if True:
            x = np.arange(len(result[1])) / rate
            f = plt.figure()
            plt.rcParams["figure.figsize"] = (8, 2.5)
            plt.plot(x, result[1], label='Power profile')
            n = 0
            print(report['time(ms)'])
            for xc in np.cumsum(result[0]['time(ms)']):
                if n == 0:
                    plt.axvline(x=xc, c='red', label='Layer transitions')
                    n = 1
                else:
                    plt.axvline(x=xc, c='red')
            #plt.axvline(x=dur/10, c='blue', label='Layer transitions')

            plt.xlabel("Time [ms]")
            plt.ylabel("Power [W]")

            plt.legend()
            plt.show()


def measure_network(optimize, execute, parse, network, framework = "tf"):
        rate = 500
        vis = False
        if framework == "tf":
            file_t = ".pb"
        elif framework == "onnx":
            file_t = ".onnx"
        else:
            print("Framework not supported!")
            return
        test_net = get_database('graphs', framework, network+file_t)
        optimize(test_net, source_fw = framework, network = network, input_shape = None , save_folder = get_database('benchmarks','tmp'))

        test_net = get_database('benchmarks','tmp', network+'.xml')
        execute_kwargs = {"xml_path": test_net, "report_dir": get_database('benchmarks','tmp'), 'device': 'MYRIAD'}
        dur, power_dir, power_file = execute(**execute_kwargs)

        test_report = get_database('benchmarks','tmp','benchmark_average_counters_report.csv')
        report = parse(test_report) 
        duration = np.sum(report['time(ms)'])
        result = processing.unite_latency_power_meas(report, power_file+'.dat', power_dir, sample_rate = rate, vis=vis, padding=100)
        print(result)
        print(len(result[1]))

        if True:
            x = np.arange(len(result[1])) / rate
            f = plt.figure()
            plt.rcParams["figure.figsize"] = (8, 2.5)
            plt.plot(x, result[1], label='Power profile')
            n = 0
            print(report['time(ms)'])
            for xc in np.cumsum(result[0]['time(ms)']):
                if n == 0:
                    plt.axvline(x=xc, c='red', label='Layer transitions')
                    n = 1
                else:
                    plt.axvline(x=xc, c='red')
            #plt.axvline(x=dur/10, c='blue', label='Layer transitions')

            plt.xlabel("Time [ms]")
            plt.ylabel("Power [W]")

            plt.legend()
            plt.show()

def measure_destruct_annette_network(optimize, execute, parse, network, config = None):
        rate = 500
        vis = False

        gen = generator.Graph_generator(network)
        if config:
            gen.add_configfile(config)

        rem_layers = []
        durations = []
        meas_durations = []
        for j, l in enumerate(reversed(gen.init_graph.topological_sort)):
            out = gen.generate_graph_from_config(100)
            select = -1
            for i, l2 in enumerate(gen.init_graph.model_spec['output_layers']):
                if len(gen.init_graph.model_spec['layers'][l2]['children']) == 0:
                    select = i

            init_graph = deepcopy(gen.graph)
            test_net = get_database('graphs', 'tf', network+'.pb')
            optimize(test_net, source_fw = 'tf', network = network, input_shape = None , save_folder = get_database('benchmarks','tmp'))

            test_net = get_database('benchmarks','tmp', network+'.xml')
            execute_kwargs = {"xml_path": test_net, "report_dir": get_database('benchmarks','tmp'), 'device': 'MYRIAD', 'name': str(j)}
            dur, power_dir, power_fil2e = execute(**execute_kwargs)
            test_report = get_database('benchmarks','tmp','benchmark_average_counters_report.csv')
            report = parse(test_report) 
            duration = np.sum(report['time(ms)'])
            durations.append(duration)
            meas_durations.append(dur)

            if select == -1:
                logging.error("All output layers have children. This should not be possible. Check your graph.")
                exit()
            name = gen.init_graph.model_spec['output_layers'][select]
            rem_layers.append(gen.init_graph.model_spec['layers'][name])
            gen.init_graph.delete_layer(name)
            print(rem_layers)
            print(durations)
            print(meas_durations)

