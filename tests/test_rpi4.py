import pytest
import sys
sys.path.append("./")
from pathlib import Path
import logging
import os

import annette.hw_modules.hw_modules.rpi4 as rpi4 
import annette.hw_modules.hw_modules.ncs2 as ncs2
import annette.benchmark.generator as generator
import annette.benchmark.matcher as matcher 
from annette import get_database

logging.basicConfig(level=logging.DEBUG)


__author__ = "Matthias Wess"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"

def test_optimize_network(network="annette_bench1.pb"):
    test_net = Path('tests','networks',network)
    ncs2.optimize_network(test_net, source_fw = "tf", network = "tmp_net", image = [1, 1, 1, 3] , input_node = "data", save_folder = "tests/tmp")

    assert True

def test_run_network(network="annette_bench1.xml"):
    test_net = Path('tests','tmp',network)
    ncs2.run_network(test_net, report_dir = "./tests/data/ncs2_ov2019")

    assert True

def test_read_ncs2_report(network="benchmark_average_counters_report"):
    report_file = get_database('benchmarks','tmp',network+'.csv')
    ncs2.read_report(report_file)
    annette_report = ncs2.r2a(report_file)
    #annette_report.to_pickle(get_database('benchmarks','tmp','annette_bench1.pkl')) 

    assert True

def test_pipeline(network="annette_bench1", shape = None):

    gen = generator.Graph_generator(network)
    gen.add_configfile("config_v6.csv")
    gen.generate_graph_from_config(401)
    test_net = get_database('graphs','tf',network+'.pb')
    ncs2.optimize_network(test_net, source_fw = "tf", network = network, image = shape , input_node = "data", save_folder = get_database('benchmarks','tmp'))
    test_net = get_database('benchmarks','tmp',network+'.xml')
    ncs2.run_network(test_net, report_dir = get_database('benchmarks','tmp'))
    test_report = get_database('benchmarks','tmp','benchmark_average_counters_report.csv')
    ncs2.read_report(test_report)
    print(gen.graph.model_spec)

    assert True

def test_matcher(network='annette_bench3', shape = None):
    gen = generator.Graph_generator(network)
    gen.add_configfile("config_v6.csv")
    gen.generate_graph_from_config(0)
    test_net = get_database('graphs','tf',network+'.pb')
    #ncs2.optimize_network(test_net, source_fw = "tf", network = network, image = shape , input_node = "data", save_folder = get_database('benchmarks','tmp'))
    #test_net = get_database('benchmarks','tmp',network+'.xml')
    #not running network
    #ncs2.run_network(test_net, report_dir = get_database('benchmarks','tmp'))
    test_report = get_database('benchmarks','tmp','benchmark_average_counters_report.csv')
    ncs2.read_report(test_report)
    print(gen.graph.model_spec) 

def test_net(network="reid",config="dummy.csv",start=0,vis=False):

    bench1 = matcher.Graph_matcher(network, config)
    result = bench1.run_bench(optimize = ncs2.inference.optimize_network, execute = ncs2.inference.run_network_new, parse = ncs2.parser.r2a, start=start, vis=vis,
        execute_kwargs = {"report_dir": get_database('benchmarks','tmp'), 'device': 'MYRIAD', 'sleep_time': 0.001})



    assert True

def test_all(network="annette_bench5",config="config_test.csv",start=0, end=None,vis=False):

    test_net = get_database('graphs','tf',network+'.pb')
    match = {
            "conv2d_0_Conv2D": {
                "name" : " Delegate/Convolution (NHWC  QC8) GEMM:0",
                "name2" : " Delegate/Convolution (NHWC  QC8) IGEMM:0",
                "conv2d_0_Relu" : "f_act",
                "Add_0" : "f_add",
            },
            "conv2d_1_Conv2D": {
                "name" : " Delegate/Convolution (NHWC  QC8) GEMM:1",
                "name2" : " Delegate/Convolution (NHWC  QC8) IGEMM:1",
                "conv2d_1_Relu" : "f_act",
                "Add_0" : "f_add",
                "Add_1" : "f_add_1",
            },
            "conv2d_2_Conv2D": {
                "name" : " Delegate/Convolution (NHWC  QC8) GEMM:3",
                "name2" : " Delegate/Convolution (NHWC  QC8) IGEMM:3",
            },
            "separable_conv2d_depthwise": {
                "name" : " Delegate/Convolution (NHWC  QC8) DWConv:5",
                "name2" : " Delegate/Convolution (NHWC  QC8) GEMM:5"
            },
            "separable_conv2d_pointwise": {
                "name" : " Delegate/Convolution (NHWC  QC8) GEMM:6",
                "name2" : " Delegate/Convolution (NHWC  QC8) IGEMM:6",
            },
            "conv2d_3_Conv2D": {
                "name" : " Delegate/Convolution (NHWC  QC8) GEMM:7",
                "name2" : " Delegate/Convolution (NHWC  QC8) IGEMM:7",
            },
            "conv2d_4_Conv2D": {
                "name" : " Delegate/Convolution (NHWC  QC8) GEMM:8",
                "name2" : " Delegate/Convolution (NHWC  QC8) IGEMM:8",
            },
            "max_pool_MaxPool": {
                "name" : " Delegate/Max Pooling (NHWC  S8):9",
                "name2" : " Delegate/Max Pooling (NHWC  S8):9",
            }
        }

    bench1 = matcher.Graph_matcher(network, config, match)
    rpi4_obj = rpi4.inference.rpi4Class(get_database('configs','rpi4.yaml'))
    bench1.run_bench(optimize = rpi4_obj.optimize_network, execute = rpi4_obj.run_network_ssh, parse = rpi4.parser.r2a, start=start, vis=vis, hardware='rpi4',
        execute_kwargs = {"model_path" : get_database('benchmarks', 'tf'), 'sleep_time': 0.001, 'use_tflite': False, 'use_pyarmnn': True,
                          'niter': 4, 'print_bool': True, 'save_dir': str(get_database('benchmarks', 'tmp'))})


    assert True

#mobilenet_v1_1.0_224.tflite
def test_rpi4_network(network="ssd_mobilenet_v1_1_metadata_1"):
    rpi4_obj = rpi4.inference.rpi4Class(get_database('configs','rpi4.yaml'))
    rpi4_obj.run_network_ssh(tflite_model=network, model_path="/home/tgrill/models/", save_dir=".", niter=4,
                        print_bool=True, sleep_time=0.0001, use_tflite=False, use_pyarmnn=True)

    assert True


def test_measure_network(network="cf_reid"):
    matcher.measure_network(optimize = ncs2.inference.optimize_network, execute = ncs2.inference.run_network_new, parse = ncs2.parser.r2a, network = network, framework = 'onnx')

def test_measure_annette_network(network="cf_reid"):
    matcher.measure_annette_network(optimize = ncs2.inference.optimize_network, execute = ncs2.inference.run_network_new, parse = ncs2.parser.r2a, network = network)

def test_measure_destruct_annette_network(network="cf_reid"):
    matcher.measure_destruct_annette_network(optimize = ncs2.inference.optimize_network, execute = ncs2.inference.run_network_new, parse = ncs2.parser.r2a, network = network, config="config_v6.csv")

def test_rpi4_r2a(network=None):
    result = rpi4.parser.r2a("report.csv")
    print(result['time(ms)'].sum())
    assert True




def main():

    #test_optimize_network()
    #test_run_network()
    #network = "benchmark_average_counters_report"
    #test_read_ncs2_report(network=network)
    #test_all(network='annette_bench1')
    #test_all(network='annette_bench2')
    #test_all(network='annette_bench3')
    #test_matcher(network='annette_bench1')
    #test_all(network='annette_bench5',config='config_v6_1.csv',start=0,vis=False)
    #test_measure_network(network='squeezenet1.0-9')
    #test_measure_annette_network(network='alexnet0')
    #test_all(network='annette_bench_1d',config='config_1d_full.csv', start=0, vis=False)
    #test_measure_destruct_annette_network(network='annette_bench5')
    #test_read_ncs2_report()

    #test_all(network='annette_bench_1d',config='config_1d_full.csv', start=0, vis=False)
    test_all(network='annette_bench5',config='config_v6_1.csv',start=0, end=10, vis=False)
    #test_net(network='reid')
    #test_rpi4_network()
    #test_rpi4_r2a()

if __name__ == '__main__':
    main()
