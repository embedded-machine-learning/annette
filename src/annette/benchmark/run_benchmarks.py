import sys
import time
import logging
import pickle
from math import ceil
import random
from threading import Thread

import torch
from torchsummary import summary
import pandas as pd

from annette import get_database
from annette.benchmark.generator import Graph_generator
from annette.benchmark.matcher_aiware import Graph_matcher, aiware_err_collection


glob_aiware_errors = []


def run_legacy(NUM_CONFIGS):
    # loop over each parameter config and export resulting TF graph:
    for conf_i in range(NUM_CONFIGS, NUM_CONFIGS+1):
        bench_graph = Graph_generator(BENCH_DIR + BENCH_NET_NAME)
        bench_graph.add_configfile(BENCH_CONFIG_FILE_NAME)

        # Use current version based on PyTorch and own accelerators:
        _, pt_graph = bench_graph.generate_graph_from_config(conf_i, framework='pytorch')

        # print all shapes of Annette graph:
        for l, params in bench_graph.graph.model_spec['layers'].items():
            in_shape = params['input_shape'] if 'input_shape' in params.keys() else 'None'
            kernel_size = params['kernel_shape'] if 'kernel_shape' in params.keys() else 'None'
            print(f'{l}{" "*(20-len(l))} : in_sh = {in_shape} ; kern = {kernel_size} ; out_sh = {params["output_shape"]}')

        # print all shapes of pytorch graph:
        model_name = BENCH_NET_NAME
        shapes = pd.read_csv(get_database('benchmarks', 'config', BENCH_CONFIG_FILE_NAME))
        shapes = shapes.loc[NUM_CONFIGS]
        inp_size = (1, 3, int(shapes['height']), int(shapes['width'])) # NCHW
        summary(pt_graph, torch.randn(inp_size),
                col_names=['input_size', 'kernel_size', 'output_size', 'mult_adds'],
                depth=5)

        from annette.hw_modules.hw_modules.aiware import optimize_network, run_network, get_layername_mapping, r2a
        optimize_network(pt_graph, inp_size, model_name)
        run_network(inp_size, model_name)
        layername_map = get_layername_mapping(bench_graph.graph)
        print(layername_map)
        r2a('./tmp/annette_convnet/estimation-results/layer_stats_0.csv', layername_map)

        # Use legacy version: tensorflow and CPU benchmark via OpenVino:
        # from annette.hw_modules.hw_modules.ncs2_ov2019 import optimize_network, run_network, read_report, r2a
        # pt_graph = bench_graph.generate_graph_from_config(conf_i, framework='tf')
        # inp_size = (1, 16, 16, 3)
        # optimize_network('/local/steinmja/repos/annette/database/graphs/tf/annette_convnet.pb', image=inp_size)
        # run_network('./tmp/annette_convnet.xml')
        # r2a('./tmp/benchmark_average_counters_report.csv')

def run_matcher(conf_indices, bench_dir, bench_name, bench_config):
    ''' Sequential version of benchmark run '''
    aiware_matcher = Graph_matcher(bench_name, bench_config, bench_dir, match=None)
    aiware_matcher.keep_tmp_data = KEEP_TMP_DATA
    aiware_matcher.run_bench(conf_indices=conf_indices)

    return aiware_matcher

def run_matcher_concurrent(conf_indices, bench_dir, bench_name, bench_config, num_threads=1):
    ''' OBACHT: Multiprocessing version '''
    from multiprocessing import Process, Manager, Lock
    glob_matchers = []
    proc_exit_codes = []
    err_lock = Lock()
    matcher_lock = Lock()
    with Manager() as man:
        matchers = man.list()
        aiware_errors = man.list()
        threads = []
        num_indices_t = ceil(len(conf_indices) / num_threads)
        for t in range(num_threads):
            t_indices = conf_indices[t*num_indices_t : (t+1)*num_indices_t]
            # Avoid "empty" threads if NUM_THREADS is not divisor of number of config indices:
            if len(t_indices) == 0:
                break

            matcher = Graph_matcher(bench_name, bench_config, bench_dir, match=None, matcher_idx=t)
            matcher.keep_tmp_data = KEEP_TMP_DATA

            thread = Process(
                target=matcher.run_bench,
                args=tuple([t_indices, matchers, matcher_lock, aiware_errors, err_lock])
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
            proc_exit_codes.append(thread.exitcode)

        while matchers:
            m = matchers.pop()
            glob_matchers.append(m)

        while aiware_errors:
            e = aiware_errors.pop()
            glob_aiware_errors.append(e)

        # clean up threads:
        while threads:
            thread = threads.pop()
            if thread.is_alive():
                thread.terminate()
            del thread

    return glob_matchers, proc_exit_codes

def merge_results(matcher_dicts):
    res = {}
    if len(matcher_dicts) > 0:
        for k in matcher_dicts[0].keys():
            res[k] = pd.concat([d[k] for d in matcher_dicts], axis=0, ignore_index=True)
            res[k] = res[k].sort_values('conf_idx', ignore_index=True)

    return res

if __name__ == '__main__':
    #################################################################################################
    # CONFIG VALUES
    #################################################################################################
    logging.basicConfig(level=logging.DEBUG)
    KEEP_TMP_DATA = False

    BENCH_DIR = 'ties/multi-layer/'
    # BENCH_DIR = 'ties/micro-kernel/'

    BENCH_NET_NAME = 'annette_convnet'
    # BENCH_NET_NAME = 'annette_fcnet'
    # BENCH_NET_NAME = 'conv2d'
    # BENCH_NET_NAME = 'avg-pool2d'
    # BENCH_NET_NAME = 'max-pool2d'
    # BENCH_NET_NAME = 'fully-connected'
    # BENCH_NET_NAME = 'dw-conv2d'
    # BENCH_NET_NAME = 'conv-transpose2d'

    BENCH_CONFIG_FILE_NAME = 'aiware/convnet.csv'
    # BENCH_CONFIG_FILE_NAME = 'aiware/compare_conv.csv'

    MAX_THREADS = 2 # 30 - 40 seems to be sweet spot for jack-jack

    # CONF_INDICES = [i for i in range(34993)] # config_v6 entries: 34993
    # CONF_INDICES = [i for i in range(66560)] # fully-connected entries
    # CONF_INDICES = [i for i in range(1024)] # aiware conv sweeps
    # CONF_INDICES = [1210] # aiware conv sweeps

    # CONF_INDICES = [2100, 2200, 2300, 2400, 2500, 4500, 4600, 4700, 4800, 5000] # random tests
    # CONF_INDICES = [i for i in range(5040)]
    CONF_INDICES = [i for i in range(599)] # convnet

    CONF_INDICES = [598] # convnet

    RUN_MODE = 'sequential' # or sequential (with(out) multi-threading)

    EPOCHS = 1
    EPOCH_LEN = ceil(len(CONF_INDICES) / EPOCHS)
    EPOCH_THREAD_REDUCE_RATE = 1 # reduce MAX_THREADS by this every epoch

    if len(sys.argv) > 1:
        PICKLE_FILE_NAME = sys.argv[1]
    else:
        logging.error('ERROR: No filename for output pickle file given as parameter!')
        sys.exit(1)

    #################################################################################################
    # MAIN CODE
    #################################################################################################
    start_time = time.time()
    print(f'Start time: {time.ctime((start_time))}')

    if RUN_MODE == 'sequential':
        # Standard mode: Run matcher successively for each config point
        exitcodes = [] # not used in single-threaded mode
        aiware_matcher = run_matcher(CONF_INDICES, BENCH_DIR, BENCH_NET_NAME, BENCH_CONFIG_FILE_NAME)
        try:
            st_result = aiware_matcher.df_out
            print(st_result)
            with open(PICKLE_FILE_NAME, 'wb') as pfile:
                pickle.dump(aiware_matcher.df_out, pfile)
        except:
            logging.error('No result df found. Skipping pickle dump.')
    else:
        try: # Load epoch result files of former runs, if existing:
            with open(PICKLE_FILE_NAME, 'rb') as f:
                out_df_epoch = [pickle.load(f)]
        except FileNotFoundError:
            out_df_epoch = []

        for epoch in range(0, EPOCHS):
            epoch_indices = CONF_INDICES[epoch*EPOCH_LEN : (epoch+1)*EPOCH_LEN]
            random.shuffle(epoch_indices) # Shuffle indices for even execution time between threads
            # Threaded mode: Run matcher with a max of MAX_THREADS parallel threads:
            aiware_matchers, exitcodes = run_matcher_concurrent(
                epoch_indices, BENCH_DIR, BENCH_NET_NAME, BENCH_CONFIG_FILE_NAME, MAX_THREADS
            )
            for i, matcher in enumerate(aiware_matchers):
                print(f'Matcher no: {i}')
                matcher_df = matcher.df_out if hasattr(matcher, 'df_out') else '   -> No df_out'
                print(matcher_df)

            out_df = merge_results(
                out_df_epoch[-1:] + [m.df_out for m in aiware_matchers if hasattr(m, 'df_out')]
            )
            print(out_df)
            with open(PICKLE_FILE_NAME, 'wb') as pfile:
                pickle.dump(out_df, pfile)

            out_df_epoch.append(out_df)
            print(f'Finished epoch {epoch}')
            MAX_THREADS -= EPOCH_THREAD_REDUCE_RATE

    end_time = time.time()
    duration_sec = end_time - start_time
    print(f'End time: {time.ctime(end_time)}')
    print(f'Duration: {duration_sec // 3600 :.0f}h {duration_sec % 3600 // 60 :.0f}m {duration_sec % 60 :0f}sec.')

    # Check for errors that occured and print them out:
    if sum(exitcodes) != 0:
        print(f'Exitcodes: {exitcodes}')

    if len(aiware_err_collection) > 0 or len(glob_aiware_errors) > 0:
        for e in aiware_err_collection:
            print(e)
        for e in glob_aiware_errors:
            print(e)
        print('There occured aiware-estimator errors!')
        # sys.exit(1)
