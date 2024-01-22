import annette.benchmark.matcher as matcher
from annette import get_database
import annette.benchmark.generator as generator
import annette.hw_modules.hw_modules.imx93 as imx93
import os
import logging
from pathlib import Path
import pytest
import sys
sys.path.append("./")

#_DATABASE_ROOT = pathlib.Path().resolve()

logging.basicConfig(level=logging.DEBUG)


__author__ = "Matthias Wess"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"


def test_optimize_network(network="annette_bench1.pb"):
    gen = generator.Graph_generator(network)
    gen.add_configfile("config_v6.csv")
    imx93_obj = imx93.inference.imx93Class(
        get_database('configs', 'imx93.yaml'))
    gen.generate_graph_from_config(401)
    test_net = get_database('graphs', 'tf', network+'.pb')
    imx93_obj.optimize_network(test_net, source_fw="tf", network=network,
                               input_node="data", save_folder=get_database('benchmarks', 'tmp'))
    assert True


def test_run_network(network="annette_bench1.xml"):
    test_net = Path('tests', 'tmp', network)
    imx93.run_network(test_net, report_dir="./tests/data/imx93_ov2019")

    assert True


def test_read_imx93_report(network="benchmark_average_counters_report"):
    report_file = get_database('benchmarks', 'tmp', network+'.csv')
    imx93.read_report(report_file)
    annette_report = imx93.r2a(report_file)
    # annette_report.to_pickle(get_database('benchmarks','tmp','annette_bench1.pkl'))

    assert True


def test_pipeline(network="annette_bench1", shape=None):

    gen = generator.Graph_generator(network)
    gen.add_configfile("config_v6.csv")
    gen.generate_graph_from_config(401)
    test_net = get_database('graphs', 'tf', network+'.pb')
    imx93.optimize_network(test_net, source_fw="tf", network=network, image=shape,
                           input_node="data", save_folder=get_database('benchmarks', 'tmp'))
    test_net = get_database('benchmarks', 'tmp', network+'.xml')
    imx93.run_network(test_net, report_dir=get_database('benchmarks', 'tmp'))
    test_report = get_database(
        'benchmarks', 'tmp', 'benchmark_average_counters_report.csv')
    imx93.read_report(test_report)
    print(gen.graph.model_spec)

    assert True


def test_matcher(network='annette_bench3', shape=None):
    gen = generator.Graph_generator(network)
    gen.add_configfile("config_v6.csv")
    gen.generate_graph_from_config(0)
    test_net = get_database('graphs', 'tf', network+'.pb')
    # imx93.optimize_network(test_net, source_fw = "tf", network = network, image = shape , input_node = "data", save_folder = get_database('benchmarks','tmp'))
    # test_net = get_database('benchmarks','tmp',network+'.xml')
    # not running network
    # imx93.run_network(test_net, report_dir = get_database('benchmarks','tmp'))
    test_report = get_database(
        'benchmarks', 'tmp', 'benchmark_average_counters_report.csv')
    imx93.read_report(test_report)
    print(gen.graph.model_spec)


def test_net(network="reid", config="dummy.csv", start=0, vis=False):

    bench1 = matcher.Graph_matcher(network, config)
    result = bench1.run_bench(optimize=imx93.inference.optimize_network, execute=imx93.inference.run_network_new, parse=imx93.parser.r2a, start=start, vis=vis,
                              execute_kwargs={"report_dir": get_database('benchmarks', 'tmp'), 'device': 'MYRIAD', 'sleep_time': 0.001})

    assert True

def test_single_network_destruct(network="annette_bench5", config="config_test.csv", start=0, end=None, vis=False):
    bench1 = matcher.Graph_matcher(network, framework='tf')
    imx93_obj = imx93.inference.imx93Class(
        get_database('configs', 'imx93.yaml'))
    bench1.run_single_network_destruct(optimize=imx93_obj.optimize_network, execute=imx93_obj.run_network_ssh, parse=imx93.parser.r2a, start=start, vis=vis, hardware='imx93',
                     execute_kwargs={"model_path": get_database('benchmarks', 'tf'), 'sleep_time': 0.001,
                                     'niter': 4, 'print_bool': True, 'save_dir': str(get_database('benchmarks', 'tmp'))})

    assert True

def test_all(network="annette_bench5", config="config_test.csv", start=0, end=None, vis=False):

    match = {
        "conv2d_0_Conv2D": {
            "name": " Delegate/Convolution (NHWC  QC8) GEMM:0",
            "name2": " Delegate/Convolution (NHWC  QC8) IGEMM:0",
            "conv2d_0_Relu": "f_act",
            "Add_0": "f_add",
        }
    }

    bench1 = matcher.Graph_matcher(network, config, match)
    imx93_obj = imx93.inference.imx93Class(
        get_database('configs', 'imx93.yaml'))
    bench1.run_bench(optimize=imx93_obj.optimize_network, execute=imx93_obj.run_network_ssh, parse=imx93.parser.r2a, start=start, vis=vis, hardware='imx93',
                     execute_kwargs={"model_path": get_database('benchmarks', 'tf'), 'sleep_time': 0.001,
                                     'niter': 4, 'print_bool': True, 'save_dir': str(get_database('benchmarks', 'tmp'))})

    assert True

# mobilenet_v1_1.0_224.tflite


def test_imx93_network(network="ssd_mobilenet_v1_1_metadata_1"):
    imx93_obj = imx93.inference.imx93Class(
        get_database('configs', 'imx93.yaml'))
    imx93_obj.run_network_ssh(tflite_model=network, model_path="/home/tgrill/models/", save_dir=".", niter=4,
                              print_bool=True, sleep_time=0.0001, use_tflite=False, use_pyarmnn=True)

    assert True


def test_measure_network(network="cf_reid"):
    imx93_obj = imx93.inference.imx93Class(
        get_database('configs', 'imx93.yaml'))
    matcher.measure_network(optimize=imx93.inference.optimize_network,
                            execute=imx93.inference.run_network_new, parse=imx93.parser.r2a, network=network, framework='onnx')


def test_measure_annette_network(network="cf_reid"):
    matcher.measure_annette_network(optimize=imx93.inference.optimize_network,
                                    execute=imx93.inference.run_network_new, parse=imx93.parser.r2a, network=network)


def test_measure_destruct_annette_network(network="cf_reid"):
    matcher.measure_destruct_annette_network(optimize=imx93.inference.optimize_network,
                                             execute=imx93.inference.run_network_new, parse=imx93.parser.r2a, network=network, config="config_v6.csv")


def test_imx_r2a(network=None):
    result = imx93.parser.r2a("report.csv")
    print(result['time(ms)'].sum())
    assert True


def main():

    # test_optimize_network(network='reid')

    # test_all(network='annette_bench_1d',config='config_1d_full.csv', start=0, vis=False)
    # test_all(network='annette_bench_conv_3',config='config_v7_2.csv',start=0, end=10, vis=False)
    # test_all(network='annette_bench_conv',config='config_v7_2.csv',start=0, end=10, vis=False)
    # test_all(network='annette_bench_conv_2',config='config_v7_2.csv',start=0, end=10, vis=False)

    #test_single_network_destruct(network='cf_reid',config='dummy.csv',start=0, end=10, vis=False)
    #test_single_network_destruct(network='mobilenet_v1',config='dummy.csv',start=0, end=10, vis=False)
    #test_all(network='yolov8n-cls-sim',
    #         config='dummy.csv', start=0, end=10, vis=False)
    #test_all(network='yolov8m',
    #         config='dummy.csv', start=0, end=10, vis=False)
    #test_all(network='yolov8l',
    #         config='dummy.csv', start=0, end=10, vis=False)
    #test_all(network='annette_bench_conv',
    #         config='mini.csv', start=0, end=10, vis=False)
    #test_all(network='yolov8n',
    #         config='dummy.csv', start=0, end=10, vis=False)
    #test_all(network='deeplabv3_mobilenet_v3_large-sim',
    #         config='dummy.csv', start=0, end=10, vis=False)
    #exit()
    #test_all(network='deeplabv3_mobilenet_v3_large-sim',
    #         config='dummy.csv', start=0, end=10, vis=False)
    #test_all(network='annette_bench_conv_padding_in',
    #         config='config_v7_2.csv', start=0, end=10, vis=False)
    #test_all(network='annette_bench_conv_padding_out',
    #         config='config_v7_2.csv', start=0, end=10, vis=False)
    test_all(network='annette_bench_conv_padding',
             config='config_v7_2.csv', start=0, end=10, vis=False)
    #test_all(network='annette_bench_conv_3',c8sho#C

    #         config='mini.csv', start=0, end=10, vis=False)
    #test_all(network='annette_bench_conv_4',
    #         config='mini.csv', start=0, end=10, vis=False)
    #test_all(network='annette_bench_conv_5',
    #         config='mini.csv', start=0, end=10, vis=False)
    test_all(network='annette_bench_conv_padded',
             config='config_v7_3_sampled.csv', start=0, end=10, vis=False)
    test_all(network='annette_bench_conv_padded',
             config='config_v7_3_stride_sampled.csv', start=0, end=10, vis=False)
    test_all(network='annette_bench_conv_padded',
             config='config_v7_3.csv', start=0, end=10, vis=False)
    #test_all(network='yolov8n',
    #         config='dummy.csv', start=0, end=10, vis=False)
    #test_all(network='yolov8n-cls-sim',
    #         config='dummy.csv', start=0, end=10, vis=False)
    #test_all(network='annette_bench_conv_3',
    #         config='config_v7_2.csv', start=0, end=10, vis=False)
    #test_all(network='annette_bench_avg',
    #         config='config_v7_2.csv', start=0, end=10, vis=False)
    #test_all(network='annette_bench_conv_avg',
    #         config='config_v7_2.csv', start=0, end=10, vis=False)
    #test_all(network='annette_bench_avg',
    #         config='config_v7_3.csv', start=0, end=10, vis=False)
    #test_all(network='annette_bench_conv_avg',
    #         config='config_v7_3.csv', start=0, end=10, vis=False)
    # test_all(network='annette_bench_conv_2',config='config_v7_3.csv',start=0, end=10, vis=False)
    # test_all(network='annette_bench_conv_4',config='config_v7_1.csv',start=0, end=10, vis=False)
    # test_net(network='reid')
    # test_imx93_network()
    # test_imx93_r2a()


if __name__ == '__main__':
    main()
