from __future__ import print_function

import json
import logging
import multiprocessing
import os
from copy import deepcopy 
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
try:
    from powerutils import measurement, processing
    POWERUTILS = True
except Exception as e:
    print(e)
    print("Could not import powerutils, power measurement will not work")
    POWERUTILS = False
    pass
import pickle

import annette.benchmark.generator as generator
from annette import get_database


class Graph_matcher():
    """Graph matcher"""

    def __init__(self, network, config='dummy.csv', match=None, framework='tensorflow'):
        # make generator object
        self.network = network
        self.gen = generator.Graph_generator(network)
        self.gen.add_configfile(config)
        self.config_name = config.split('.')[0]
        self.framework = framework
        self.match = match
        self.df = {}
        self.column_names = ['width', 'height', 'channels', 'filters',
                             'k_height', 'k_width', 'k_stride', 'pool_h',
                             'pool_w', 'pool_stride', 'pad', 'batch_size',
                             'f_pool', 'f_bias', 'f_add', 'f_add_1',
                             'f_concat', 'f_act', 'f_pointwise',
                             'time(ms)', 'mean(W)', 'mult(mJ)']
        # make layer list
        print(self.gen.graph.model_spec['layers'])

    def run_network(self, hardware, execute, execute_kwargs):
        # gather power data into .dat file
        # pm = measurement.power_measurement(sampling_rate=rate*1000, data_dir="./tmp", max_duration=60)
        # pm_kwargs = {"model_name": "tmpmodel"}
        # pm.start_gather(pm_kwargs) # power data aquisation

        # execute_kwargs = {"xml_path": test_net, "report_dir": get_database('benchmarks','tmp'), 'device': 'MYRIAD', 'sleep_time': 0.001}
        # execute(test_net, report_dir = get_database('benchmarks','tmp'))
        # dur, power_dir, power_file = execute(**execute_kwargs)
        def run_network_wrapped(_, kwargs):
            report_dir = execute(**kwargs)
            kwargs['report_file'] = report_dir

        if hardware in ['ncs']:
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            return_dict.update(execute_kwargs)
            p = multiprocessing.Process(
                target=run_network_wrapped, args=(0, return_dict))
            p.start()
            p.join(60)
            if p.is_alive():
                print("Execution seem stuck!")
                p.terminate()
                p.join()
                return False
            elif p.exitcode != 0:
                print("Execution failed with exitcode: {}".format(p.exitcode))
                return False
            else:
                print("Execution run successfully")
                # end power measurement
                # pm.end_gather(True)  # stop the power measurement
                # store dat file
                # dat_file_path = pm.dat_filepath # filepath of the dat file with the measurement
            print(f"KWARGS = {return_dict}")
            report_file = return_dict['report_file']
        else:
            print(f"KWARGS = {execute_kwargs}")
            report_file = execute(**execute_kwargs)
        return report_file

    def run_single_network(self, optimize=None, execute=None, parse=None, store=5,
                           hardware='ncs2', start=0, end=None, vis=False,
                           execute_kwargs={}):
        self.bench_name = hardware+'_single'

        file_format, file_folder = self.generate_file_names(hardware)
        self.gen.generate_graph_from_config(0, format=file_format, framework=self.framework)
        test_net = get_database('graphs', file_folder, self.network+f'.{file_format}')
        if optimize is not None:
            execute_kwargs.update(optimize(test_net, source_fw=self.framework, network=self.network,
                                    input_shape=None, save_folder=get_database('benchmarks', 'tmp')))

        logging.debug(f"Optimization arguments {execute_kwargs}")
        # test_net = get_database('benchmarks','tmp', self.network+'.xml')

        report_file = self.run_network(hardware, execute, execute_kwargs)
        if report_file is False:
            return False

        report = parse(report_file)
        print(report)
        duration = np.sum(report['time(ms)'])

    def run_single_network_destruct(self, optimize=None, execute=None, parse=None, store=5,
                           hardware='ncs2', start=0, end=None, vis=False,
                           execute_kwargs={}):

        rem_layers = []
        layer_names = []
        meas_durations = []

        file_format, file_folder = self.generate_file_names(hardware)
        test_net = get_database('graphs', file_folder, self.network+f'.{file_format}')

        for j, l in enumerate(reversed(self.gen.init_graph.topological_sort)):
            net_destruct = f'destruct/{self.network}_destruct_{j}'
            self.gen.init_graph.model_spec['name'] = net_destruct
            folder = get_database('graphs', file_folder, 'destruct')
            folder.mkdir(
                parents=True, exist_ok=True)
            test_net_destruct = get_database('graphs', file_folder, net_destruct+f'.{file_format}')
            self.gen.generate_graph_from_config(0, format=file_format,
                                                framework=self.framework,
                                                noreset=True)
            select = -1
            for i, l2 in enumerate(self.gen.init_graph.model_spec['output_layers']):
                if len(self.gen.init_graph.model_spec['layers'][l2]['children']) == 0:
                    select = i
            # break if input layer equals output layer
            if self.gen.init_graph.model_spec['output_layers'] == self.gen.init_graph.model_spec['input_layers']:
                break

            #execute_kwargs.update(optimize(test_net_destruct, source_fw="tf", network=net_destruct,
            #                    input_shape=None, save_folder=get_database('benchmarks', 'tmp')))

            logging.debug(f"Optimization arguments {execute_kwargs}")

            def run_network_wrapped(dummy, kwargs):
                report_dir = execute(**kwargs)
                kwargs['report_file'] = report_dir

            if optimize is not None:
                execute_kwargs.update(optimize(test_net_destruct, source_fw=self.framework, network=net_destruct,
                                        input_shape=None, save_folder=get_database('benchmarks', 'tmp')))

            logging.debug(f"Optimization arguments {execute_kwargs}")
            # test_net = get_database('benchmarks','tmp', self.network+'.xml')

            report_file = self.run_network(hardware, execute, execute_kwargs)
            if report_file is False:
                print("this")
                return False

            report = parse(report_file)
            print(report)
            duration = np.sum(report['time(ms)'])
            print(duration)

            if select == -1:
                logging.error(
                    "All output layers have children. This should not be possible. Check your graph.")
                exit()

            name = self.gen.init_graph.model_spec['output_layers'][select]
            rem_layers.append(self.gen.init_graph.model_spec['layers'][name])
            meas_durations.append(duration)
            self.gen.init_graph.delete_layer(name)
            layer_names.append(name)

            print(rem_layers)
            print(meas_durations)
            print(layer_names)
        print(rem_layers)
        print(meas_durations)
        print(layer_names)
        # write to dataframe
        df = pd.DataFrame()
        df['layer'] = layer_names
        df['duration'] = meas_durations
        df['layer_type'] = [l['type'] for l in rem_layers]

        # store dataframe
        try:
            os.makedirs(get_database(
                'benchmarks', hardware, 'destruct'))
        except Exception as e:
            logging.debug(e)
            pass
        df.to_pickle(get_database(
            'benchmarks', hardware, 'destruct', self.network+'.p'))

    def run_bench(self, optimize=None, execute=None, parse=None, store=5,
                  hardware='ncs2', start=0, end=None, vis=False,
                  execute_kwargs={}):
        self.bench_name = hardware+'_'+self.config_name
        config_len = len(self.gen.config)
        self.total_df = deepcopy(self.gen.config)
        self.total_df['time(ms)'] = np.nan

        assert (start >= config_len,
                f"Selected starting number {start} \
                larger than config length {config_len}")
        if end is None:
            end = config_len
        else:
            assert end <= config_len, f"Selected end number {end} larger than config length {config_len}"

        # load current counter if file exists
        try:
            with open(get_database('benchmarks', self.bench_name, self.network,
                                   'current.txt'), 'r') as outfile:
                start = int(outfile.read()) 
            # load self.df
        except Exception as e:
            logging.debug(e)
            start = 0
            pass
        try:
            with open(get_database('benchmarks', self.bench_name, self.network,
                                   'df.p'), 'rb') as outfile:
                self.df = pickle.load(outfile)
        except Exception as e:
            logging.debug(e)
            pass
        try:
            with open(get_database('benchmarks', self.bench_name, self.network,
                                   'total.pkl'), 'rb') as outfile:
                self.total_df = pickle.load(outfile)
        except Exception as e:
            logging.debug(e)
            pass

        for i in range(start, end):
            print('-'*60)
            print('Running Config %i of %i' % (i, end))
            print('-'*60)
            
            file_format, file_folder = self.generate_file_names(hardware)
            self.gen.generate_graph_from_config(i, format=file_format, framework=self.framework)
            test_net = get_database('graphs', file_folder, self.network+f'.{file_format}')
            if optimize is not None:
                execute_kwargs.update(optimize(test_net, source_fw=self.framework, network=self.network,
                                      input_shape=None, save_folder=get_database('benchmarks', 'tmp')))

            logging.debug(f"Optimization arguments {execute_kwargs}")
            # test_net = get_database('benchmarks','tmp', self.network+'.xml')

            report_file = self.run_network(hardware, execute, execute_kwargs)
            if report_file is False:
                continue
                break

            report = parse(report_file)
            # print(report)
            # apply median to report['time(ms)'] for each row since it is a list
            report['time(ms)'] = report['time(ms)'].apply(lambda x: np.min(x))
            duration = np.sum(report['time(ms)'])
            self.total_df.at[i, 'time(ms)'] = duration
            print(self.total_df)

            # total_result = processing.extract_power_profile(pm.dat_filename, pm.data_dir, duration, sample_rate = rate)

            # if power measurement is not available, use dummy data
            # result = processing.unite_latency_power_meas(report, 'test_infmod.dat', 'tmp/', sample_rate = rate, padding=100, vis=vis)

            # print(type(report))
            result = report

            try:
                os.makedirs(get_database(
                    'benchmarks', self.bench_name, self.network))
            except Exception as e:
                logging.debug(e)
                pass
            # Store total time
            with open(get_database('benchmarks', self.bench_name, self.network, 'total.pkl'), 'wb') as outfile:
                pickle.dump(self.total_df, outfile)
            # store current counter config
            with open(get_database('benchmarks', self.bench_name, self.network, 'current.txt'), 'w') as outfile:
                # store value of counter
                outfile.write(str(i))

            if self.match is not None:
                self.match_and_add(self.gen.graph, result)
                if i % store == 0 and i > store-1 or i <= config_len-1:
                    # print(i)
                    # store self.df
                    with open(get_database('benchmarks', self.bench_name, self.network, 'df.p'), 'wb') as outfile:
                        pickle.dump(self.df, outfile)
                    for key, v in self.df_out.items():
                        v.to_pickle(get_database(
                            'benchmarks', self.bench_name, self.network, key+'.p'))
                    

        return result

    def generate_file_names(self, hardware):
        """Generate file names for different hardware"""
        if hardware in ['rpi4', 'imx93', 'imx8', 'gap9']:
            file_format = 'tflite'
            file_folder = 'tf'
        elif hardware in ['xavier']:
            file_format = 'onnx'
            file_folder = 'onnx'
        else:
            file_format = 'pb'
            file_folder = 'tf'
        return file_format, file_folder

    def match_and_add(self, graph, report):
        # compares graph with report
        logging.debug(report)
        logging.debug(graph.model_spec)
        missing_layers = []
        for l_name, l_attr in self.gen.graph.model_spec['layers'].items():
            # check if layer was executed
            report_name = None
            logging.debug("L_name: %s" % l_name)
            # print([l for l in report['name'].to_numpy(dtype=str) if l.find(l_name) != 1])
            r_name = l_name
            r2_name = l_name
            if l_name in self.match.keys():
                if 'name' in self.match[l_name].keys():
                    r_name = self.match[l_name]['name']
                if 'name2' in self.match[l_name].keys():
                    r2_name = self.match[l_name]['name2']
            
            if r_name in report['name'].to_numpy(dtype=str):
                report_name = r_name
                self.gen.graph.model_spec['layers'][l_name]['report_name'] = report_name
            if r2_name in report['name'].to_numpy(dtype=str):
                report_name = r2_name
                self.gen.graph.model_spec['layers'][l_name]['report_name'] = report_name
            elif 'Placeholder' in l_name:
                # print(report)
                if 'NaN' in report['name'].to_numpy():
                    report_name = 'NaN'
                    self.gen.graph.model_spec['layers'][l_name]['report_name'] = report_name
            elif len([l for l in report['name'].to_numpy(dtype=str) if l.find(l_name) != 1]) < 5:
                report_name = [l for l in report['name'].to_numpy(
                    dtype=str) if l.find(l_name) != 1]
                report_name = report_name[0]
                self.gen.graph.model_spec['layers'][l_name]['report_name'] = report_name
                logging.debug("layer %s found " % l_name)
                logging.debug("layer %s found " % report_name)
            else:
                missing_layers.append(l_name)
                logging.debug("layer %s not found " % l_name)
        logging.debug("Missing layers %s" % missing_layers)

        # TODO make this a separate function?
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
                    tmp['width'] = l_attr['input_shape'][1]
                except Exception as e:
                    logging.debug(e)
                    pass
                try:
                    tmp['height'] = l_attr['input_shape'][2]
                except Exception as e:
                    logging.debug(e)
                    pass
                try:
                    tmp['channels'] = l_attr['input_shape'][3]
                except Exception as e:
                    logging.debug(e)
                    pass
                if l_attr['type'] == 'DataInput':
                    tmp.update({
                        'width': l_attr['output_shape'][1],
                        'height': l_attr['output_shape'][2],
                        'channels': l_attr['output_shape'][3],
                    })
                elif l_attr['type'] == 'Conv':
                    tmp.update({
                        'filters': l_attr['output_shape'][3],
                        'k_width': l_attr['kernel_shape'][0],
                        'k_height': l_attr['kernel_shape'][1],
                        'k_stride': l_attr['strides'][1],
                    })
                elif l_attr['type'] == 'DepthwiseConv':
                    tmp.update({
                        'filters': l_attr['output_shape'][3],
                        'k_width': l_attr['kernel_shape'][0],
                        'k_height': l_attr['kernel_shape'][1],
                        'k_stride': l_attr['strides'][1],
                    })
                elif l_attr['type'] == 'MatMul':
                    tmp.update({
                        'filters': l_attr['output_shape'][1],
                    })
                elif l_attr['type'] == 'Concat':
                    def try_else_0(x,y):
                        try:
                            return x[y]
                        except Exception as e:
                            logging.debug(e)
                            return 0
                    tmp.update({
                        'width': try_else_0(l_attr['output_shape'],1),
                        'height': try_else_0(l_attr['output_shape'],2),
                        'channels': try_else_0(l_attr['output_shape'],3)
                    })

                def add_to_tmp(report, report_name, key):
                    return report[key][report['name'] == report_name].to_numpy(dtype=str)

                tmp['time(ms)'] = add_to_tmp(report, report_name, 'time(ms)')
                if 'mult(mJ)' in report.keys():
                    tmp['mult(mJ)'] = add_to_tmp(
                        report, report_name, 'mult(mJ)')
                if 'mem(mJ)' in report.keys():
                    tmp['mean(W)'] = add_to_tmp(report, report_name, 'mean(W)')
                report = report[report['name'] != report_name]

                logging.debug(self.match)
                if l_name in self.match.keys():
                    for l_key in self.match[l_name]:
                        if l_key in missing_layers:
                            tmp[self.match[l_name][l_key]] = True
                fill = [tmp.get(x) for x in self.column_names]
                self.df[l_attr['type']].append(fill)
                self.df[l_name].append(fill)
                # Remove Layer for dataframe

        self.df_out = {}

        for l_type, l_df in self.df.items():
            self.df_out[l_type] = pd.DataFrame(
                self.df[l_type], columns=self.column_names)
            logging.debug(l_type)
            logging.debug(self.df_out[l_type])

        if len(report) > 0:
            logging.debug("Left Layers: %s" % report)
        else:
            logging.debug("No left layers")

def measure_annette_network(optimize, execute, parse, network, hardware):
    rate = 500
    vis = False

    gen = generator.Graph_generator(network)
    gen.config = None
    gen.generate_graph_from_config(0)
    test_net = get_database('graphs', 'tf', network+'.pb')

    optimize(test_net, source_fw="tf", network=network,
             input_shape=None, save_folder=get_database('benchmarks', 'tmp'))

    test_net = get_database('benchmarks', 'tmp', network+'.xml')
    execute_kwargs = {"xml_path": test_net, "report_dir": get_database(
        'benchmarks', 'tmp'), 'device': hardware}
    execute(**execute_kwargs)

    power_file = 'test_infmod.dat'
    power_dir = 'tmp/'

    test_report = get_database(
        'benchmarks', 'tmp', 'benchmark_average_counters_report.csv')
    report = parse(test_report)
    duration = np.sum(report['time(ms)'])
    if POWERUTILS is True:
        result = processing.unite_latency_power_meas(
            report, power_file, power_dir, sample_rate=rate, vis=vis, padding=200)
        print(result)
        print(len(result[1]))

        if True:
            x = np.arange(len(result[1])) / rate
            plt.figure()
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
            # plt.axvline(x=dur/10, c='blue', label='Layer transitions')

            plt.xlabel("Time [ms]")
            plt.ylabel("Power [W]")

            plt.legend()
            plt.show()


def measure_network(optimize, execute, parse, network, framework="tf"):
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
    optimize(test_net, source_fw=framework, network=network,
             input_shape=None, save_folder=get_database('benchmarks', 'tmp'))

    test_net = get_database('benchmarks', 'tmp', network+'.xml')
    execute_kwargs = {"xml_path": test_net, "report_dir": get_database(
        'benchmarks', 'tmp'), 'device': 'MYRIAD'}
    dur, power_dir, power_file = execute(**execute_kwargs)

    test_report = get_database(
        'benchmarks', 'tmp', 'benchmark_average_counters_report.csv')
    report = parse(test_report)
    duration = np.sum(report['time(ms)'])
    if POWERUTILS is True:
        result = processing.unite_latency_power_meas(
            report, power_file+'.dat', power_dir, sample_rate=rate, vis=vis, padding=100)
        print(result)
        print(len(result[1]))

        if True:
            x = np.arange(len(result[1])) / rate
            plt.figure()
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
            # plt.axvline(x=dur/10, c='blue', label='Layer transitions')

            plt.xlabel("Time [ms]")
            plt.ylabel("Power [W]")

            plt.legend()
            plt.show()


def measure_destruct_annette_network(optimize, execute, parse, network, config=None, hardware='ncs2', power=False, rate=500, port=5, execute_kwargs={}):
    vis = False

    gen = generator.Graph_generator(network)
    if config:
        gen.add_configfile(config)

    rem_layers = []
    durations = []
    meas_durations = []
    if power is True:
        get_database('benchmarks', hardware, 'destruct').mkdir(
            parents=True, exist_ok=True)
        dir_path = get_database('benchmarks', hardware)
        dir_str = str(dir_path)
    for j, l in enumerate(reversed(gen.init_graph.topological_sort)):
        net_destruct = f'destruct/{network}_destruct_{j}'
        gen.init_graph.model_spec['name'] = net_destruct
        get_database('graphs', 'tf', 'destruct').mkdir(
            parents=True, exist_ok=True)
        out = gen.generate_graph_from_config(100, format='tf', noreset=True)
        select = -1
        for i, l2 in enumerate(gen.init_graph.model_spec['output_layers']):
            if len(gen.init_graph.model_spec['layers'][l2]['children']) == 0:
                select = i
        # break if input layer equals output layer
        if gen.init_graph.model_spec['output_layers'] == gen.init_graph.model_spec['input_layers']:
            break
        init_graph = deepcopy(gen.graph)
        test_net = get_database('graphs', 'tf', net_destruct+'.pb')

        execute_kwargs.update(optimize(test_net, source_fw="tf", network=net_destruct,
                              input_shape=None, save_folder=get_database('benchmarks', 'tmp')))

        logging.debug(f"Optimization arguments {execute_kwargs}")

        def run_network_wrapped(dummy, kwargs):
            report_dir = execute(**kwargs)
            kwargs['report_file'] = report_dir

        if power is True and POWERUTILS is True:
            print('start power measurement')
            print(f'{dir_str}, {net_destruct}')
            pm = measurement.power_measurement(
                sampling_rate=rate*1000, data_dir=f'{dir_str}', max_duration=150, port=port)
            pm_kwargs = {"model_name": net_destruct}
            pm.start_gather(pm_kwargs)  # power data aquisation

        if hardware in ['ncs']:
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            return_dict.update(execute_kwargs)
            # start power measurement
            p = multiprocessing.Process(
                target=run_network_wrapped, args=(0, return_dict))
            p.start()
            p.join(60)
            if p.is_alive():
                print("Execution seem stuck!")
                p.terminate()
                p.join()
                break
            elif p.exitcode != 0:
                print("Execution failed with exitcode: {}".format(p.exitcode))
                break
            else:
                print("Execution run successfully")
                # end power measurement
                # pm.end_gather(True)  # stop the power measurement
                # store dat file
                # dat_file_path = pm.dat_filepath # filepath of the dat file with the measurement
            print(f"KWARGS = {return_dict}")
            report_file = return_dict['report_file']
        else:
            print(f"KWARGS = {execute_kwargs}")
            report_file = execute(**execute_kwargs)
            print("done")
        if power is True and POWERUTILS is True:
            pm.end_gather(True)  # power data aquisation end

        if select == -1:
            logging.error(
                "All output layers have children. This should not be possible. Check your graph.")
            exit()
        name = gen.init_graph.model_spec['output_layers'][select]
        rem_layers.append(gen.init_graph.model_spec['layers'][name])
        gen.init_graph.delete_layer(name)


def measure_destruct_annette_tflite(optimize, execute, parse, network, config=None, hardware='imx8', power=False, rate=500, port=5, niter=100):
    vis = False

    gen = generator.Graph_generator(network)
    if config:
        gen.add_configfile(config)

    rem_layers = []
    durations = []
    meas_durations = []
    if power is True and POWERUTILS is True:
        get_database('benchmarks', hardware, 'destruct').mkdir(
            parents=True, exist_ok=True)
        dir_path = get_database('benchmarks', hardware)
        dir_str = str(dir_path)
    for j, l in enumerate(reversed(gen.init_graph.topological_sort)):
        net_destruct = f'destruct/{network}_destruct_{j}'
        gen.init_graph.model_spec['name'] = net_destruct
        get_database('graphs', 'tf', 'destruct').mkdir(
            parents=True, exist_ok=True)
        out = gen.generate_graph_from_config(
            100, format='tflite', noreset=True)
        select = -1
        for i, l2 in enumerate(gen.init_graph.model_spec['output_layers']):
            if len(gen.init_graph.model_spec['layers'][l2]['children']) == 0:
                select = i
        # break if input layer equals output layer
        if gen.init_graph.model_spec['output_layers'] == gen.init_graph.model_spec['input_layers']:
            break

        init_graph = deepcopy(gen.graph)
        test_net = get_database('graphs', 'tf', f'{net_destruct}.tflite')
        optimize(test_net, source_fw='tflite', network=net_destruct,
                 input_shape=None, save_folder=get_database('benchmarks', 'tmp'))

        test_net = get_database('graphs', 'tf', net_destruct+'.tflite')
        execute_kwargs = {"tflite_model": net_destruct, "model_path": get_database('graphs', 'tf'), "save_dir": get_database(
            'benchmarks', 'tmp'), 'niter': niter, 'print_bool': False, 'sleep_time': 0.1}
        # start power measurement
        if power is True and POWERUTILS is True:
            print('start power measurement')
            print(f'{dir_str}, {net_destruct}')
            pm = measurement.power_measurement(
                sampling_rate=rate*1000, data_dir=f'{dir_str}', max_duration=150, port=port)
            pm_kwargs = {"model_name": net_destruct}
            pm.start_gather(pm_kwargs)  # power data aquisation

        test_report = execute(**execute_kwargs)
        if power is True and POWERUTILS is True:
            pm.end_gather(True)  # power data aquisation end

        print(test_report)
        report = parse(test_report)
        duration = np.sum(report['time(ms)'])
        durations.append(duration)
        dur = duration
        meas_durations.append(dur)

        if select == -1:
            logging.error(
                "All output layers have children. This should not be possible. Check your graph.")
            exit()
        name = gen.init_graph.model_spec['output_layers'][select]
        rem_layers.append(gen.init_graph.model_spec['layers'][name])
        rem_layers[-1]['name'] = name
        gen.init_graph = gen.init_graph.delete_layer(name)
        print("Removed:", rem_layers)
        print(durations)
        print(meas_durations)
    # make directory with pathlib
    Path(get_database('benchmarks', hardware)).mkdir(
        parents=True, exist_ok=True)

    # store results
    with open(get_database('benchmarks', hardware, network+'_destruct.json'), 'w') as fp:
        json.dump({'layers': rem_layers, 'durations': durations,
                  'meas_durations': meas_durations}, fp)
