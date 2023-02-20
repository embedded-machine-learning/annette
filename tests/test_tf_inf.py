import pytest
import sys
sys.path.append("./")
from pathlib import Path
import logging
import os

import annette.hw_modules.hw_modules.tf_basic as tf_basic
import annette.benchmark.generator as generator
import annette.benchmark.matcher as matcher 
from annette import get_database

level = logging.DEBUG
logger = logging.getLogger()
logger.setLevel(level)


__author__ = "Matthias Wess"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"

def test_optimize_network(network="annette_bench1.pb"):
    test_net = Path('tests','networks',network)
    test = tf_basic.optimize_network(test_net, source_fw = "tf", network = "tmp_net", image = [1, 1, 1, 3] , input_node = "data", save_folder = "tests/tmp")
    print(test)

    assert True

def test_run_network(network="annette_bench5"):
    test_net = get_database('graphs','tf',network+'.pb')
    tf_basic.run_network(test_net)

    assert True

def test_read_report(network="timeline_01"):
    report_file = get_database('benchmarks','tmp',network+'.json')
    tf_basic.read_report(report_file)
    annette_report = tf_basic.r2a(report_file)
    #annette_report.to_pickle(get_database('benchmarks','tmp','annette_bench1.pkl')) 

    assert True

def test_pipeline(network="annette_bench5", shape = None):

    gen = generator.Graph_generator(network)
    gen.add_configfile("config_v6.csv")
    gen.generate_graph_from_config(401)
    test_net = get_database('graphs','tf',network+'.pb')
    tf_basic.optimize_network(test_net, source_fw = "tf", network = network, input_shape = shape , input_node = "data", save_folder = get_database('benchmarks','tmp'))
    tf_basic.run_network(test_net, save_folder = get_database('benchmarks','tmp'), network = network)
    report_file = get_database('benchmarks','tmp',network+'.json')
    tf_basic.read_report(report_file)
    annette_report = tf_basic.r2a(report_file)
    logging.debug(annette_report)

    assert True

"""
def test_matcher(network='annette_bench5', shape = None):
    gen = generator.Graph_generator(network)
    gen.add_configfile("config_v6.csv")
    gen.generate_graph_from_config(401)
    test_net = get_database('graphs','tf',network+'.pb')
    #ncs2.optimize_network(test_net, source_fw = "tf", network = network, image = shape , input_node = "data", save_folder = get_database('benchmarks','tmp'))
    #test_net = get_database('benchmarks','tmp',network+'.xml')
    #not running network
    #ncs2.run_network(test_net, report_dir = get_database('benchmarks','tmp'))
    test_report = get_database('benchmarks','tmp','benchmark_average_counters_report.csv')
    tf_basic.read_report(test_report)
    print(gen.graph.model_spec) 
"""

def test_all(network="annette_bench5",config="config_v6_1.csv"):

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

    bench1 = matcher.Graph_matcher(network, config, match)
    bench1.run_bench(optimize = tf_basic.optimize_network, execute = tf_basic.run_network, parse = tf_basic.r2a, hardware='tf_basic', end = None)

    for key, v in bench1.df_out.items():
        print(key)
        print(v)

    assert True


def main():

    #test_optimize_network()
    #test_run_network()
    #test_pipeline()
    #network = "benchmark_average_counters_report"
    #test_read_ncs2_report(network=network)
    #test_all(network='annette_bench5')
    test_all(network='annette_bench0',config='conv2d_finesweep.csv')
    #test_run_inference(network='annette_bench1')
    #test_read_report(network='timeline_01')
    #test_all(network='annette_bench2')
    #test_all(network='annette_bench3')
    #test_matcher(network='annette_bench1')
    #test_all(network='annette_bench5')
    #test_read_report()

if __name__ == '__main__':
    main()
