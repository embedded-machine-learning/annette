import pytest
import sys
sys.path.append("./")
from pathlib import Path
import logging
import os

import annette.hw_modules.hw_modules.ncs2_ov2019 as ncs2
import annette.benchmark.generator as generator
from annette import get_database

print(ncs2.__dict__)

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
    annette_report.to_pickle(get_database('benchmarks','tmp','annette_bench1.pkl')) 

    assert True

def test_all(network="annette_bench1", shape = None):

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
    gen.generate_graph_from_config(401)
    test_net = get_database('graphs','tf',network+'.pb')
    #ncs2.optimize_network(test_net, source_fw = "tf", network = network, image = shape , input_node = "data", save_folder = get_database('benchmarks','tmp'))
    #test_net = get_database('benchmarks','tmp',network+'.xml')
    #not running network
    #ncs2.run_network(test_net, report_dir = get_database('benchmarks','tmp'))
    test_report = get_database('benchmarks','tmp','benchmark_average_counters_report.csv')
    ncs2.read_report(test_report)
    print(gen.graph.model_spec) 



def main():

    #test_optimize_network()
    #test_run_network()
    #network = "benchmark_average_counters_report"
    #test_read_ncs2_report(network=network)
    #test_all(network='annette_bench1')
    #test_all(network='annette_bench2')
    #test_all(network='annette_bench3')
    test_matcher(network='annette_bench1')
    test_read_ncs2_report()

if __name__ == '__main__':
    main()
