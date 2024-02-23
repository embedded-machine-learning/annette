import pytest
import sys
sys.path.append("./")
from pathlib import Path
import logging
import os

import annette.hw_modules.hw_modules.openvino as openvino
import annette.benchmark.generator as generator
import annette.benchmark.matcher as matcher 
from annette import get_database

logging.basicConfig(level=logging.DEBUG)


__author__ = "Matthias Wess"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"

def test_optimize_network(network="annette_bench1.pb"):
    openvino_obj = openvino.inference.openvinoClass(
        get_database('configs', 'config.yaml'))
    test_net = Path('tests','networks',network)
    openvino_obj.optimize_network(test_net, source_fw = "tf", network = "tmp_net", image = [1, 1, 1, 3] , input_node = "data", save_folder = "tests/tmp")
    assert True

def test_run_network(network="annette_bench1.xml"):
    openvino_obj = openvino.inference.openvinoClass(
        get_database('configs', 'config.yaml'))
    test_net = Path('tests','tmp',network)
    openvino.run_network(test_net, report_dir = "./tests/data/openvino")
    assert True

def test_pipeline(network="annette_bench1", shape = None):
    openvino_obj = openvino.inference.openvinoClass(
        get_database('configs', 'config.yaml'))
    gen = generator.Graph_generator(network)
    gen.add_configfile("config_v6.csv")
    gen.generate_graph_from_config(401)
    test_net = get_database('graphs','tf',network+'.pb')
    openvino_obj.optimize_network(test_net, source_fw = "tf", network = network, image = shape , input_node = "data", save_folder = get_database('benchmarks','tmp'))
    test_net = get_database('benchmarks','tmp',network+'.xml')
    openvino_obj.run_network(test_net, report_dir = get_database('benchmarks','tmp'))
    test_report = get_database('benchmarks','tmp','benchmark_average_counters_report.csv')
    openvino_obj.read_report(test_report)
    print(gen.graph.model_spec)

    assert True

def test_all(network="annette_bench5",config="config_test.csv",start=0,vis=False, match = None):
    openvino_obj = openvino.inference.openvinoClass(
        get_database('configs', 'config.yaml'))
    match = {
            "conv2d_0_Conv2D": {
                "conv2d_0_Relu" : "f_act",
                "Add_0" : "f_add",
            },
            "conv2d_1_Conv2D": {
                "conv2d_1_Relu" : "f_act",
                "Add_0" : "f_add",
                "Add_1" : "f_add_1",
            },
            "conv2d_2_Conv2D": {
                "conv2d_2_Relu" : "f_act",
                "Add_1" : "f_add_1",
            },
            "conv2d_3_Conv2D": {
                "conv2d_3_Relu" : "f_act",
                "concat" : "f_concat"
            },
            "conv2d_4_Conv2D": {
                "conv2d_4_Relu" : "f_act",
                "max_pool_MaxPool" : "f_pool",
                "concat" : "f_concat"
            },
            "fully_conn_0_MatMul": {
                "fully_conn_0_Relu" : "f_act",
            },
            "fully_conn_1_MatMul": {
                "fully_conn_1_Softmax" : "f_act",
            }
        }

    match = None

    bench1 = matcher.Graph_matcher(network, config, match)
    bench1.run_bench(optimize = openvino_obj.optimize_network, execute = openvino_obj.run_network, parse = openvino.parser.r2a, start = start, vis = vis, hardware = 'ov_cpu',
        execute_kwargs = {"report_dir": get_database('benchmarks','tmp'), 'device': 'CPU', 'sleep_time': 0.001})

    try:
        for key, v in bench1.df_out.items():
            print(key)
            print(v)
    except:
        print("No data")
        pass

    assert True


def test_measure_network(network="cf_reid"):
    openvino_obj = openvino.inference.openvinoClass(
        get_database('configs', 'config.yaml'))
    matcher.measure_network(optimize = openvino_obj.optimize_network, execute = openvino_obj.run_network, parse = openvino.parser.r2a, network = network, hardware = 'ov_cpu')

def test_measure_annette_network(network="cf_reid"):
    openvino_obj = openvino.inference.openvinoClass(
        get_database('configs', 'config.yaml'))
    matcher.measure_annette_network(optimize = openvino_obj.optimize_network, execute = openvino_obj.run_network, parse = openvino.parser.r2a, network = network, hardware = 'ov_cpu')

def test_measure_destruct_annette_network(network="cf_reid"):
    openvino_obj = openvino.inference.openvinoClass(
        get_database('configs', 'config.yaml'))
    matcher.measure_destruct_annette_network(optimize = openvino_obj.optimize_network, execute = openvino_obj.run_network, parse = openvino.parser.r2a, network = network, config="config_v6.csv")

def main():

    #test_optimize_network()
    #test_run_network()
    #network = "benchmark_average_counters_report"
    #test_read_openvino_report(network=network)
    #test_all(network='annette_bench1')
    #test_all(network='annette_bench2')
    #test_all(network='annette_bench3')
    #test_matcher(network='annette_bench1')
    test_all(network='annette_bench_conv_padded',config='config_v7_3_sampled.csv', start=0, vis=False)
    #test_all(network='cf_inceptionv4',config='dummy.csv', start=0, vis=False)
    #test_all(network='annette_bench5',config='config_v6_1.csv', start=0, vis=False)
    #test_all(network='annette_bench_1d',config='config_1d_full.csv', start=0, vis=False)
    #test_measure_network(network='squeezenet1.0-9')
    #test_measure_annette_network(network='alexnet0')
    #test_measure_destruct_annette_network(network='annette_bench5')
    #test_read_openvino_report()

if __name__ == '__main__':
    main()
