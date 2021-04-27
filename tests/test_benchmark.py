import pytest
import sys
sys.path.append("./")
from pathlib import Path
import logging
import os

logging.basicConfig(level=logging.DEBUG)

from annette.graph import MMGraph
from annette.graph import AnnetteGraph
import annette.benchmark.generator as generator
import annette.benchmark.matcher as matcher 

import annette.hw_modules.hw_modules.ncs2_ov2019 as ncs2
from annette import get_database

__author__ = "Matthias Wess"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"


def test_annette_to_model(network="cf_reid"):
    json_file = Path('database','graphs','annette',network+'.json')
    annette_graph = AnnetteGraph(network, json_file)

    # execute the function under test
    generator.generate_tf_model(annette_graph)
    assert True

def test_annette_to_model_from_config(network="cf_reid"):

    gen = generator.Graph_generator(network)
    print(gen.__dict__)
    gen.add_configfile("config_v6.csv")
    gen.generate_graph_from_config(4002)

    assert True

def test_compute_dims(network="annette_bench1"):
    gen = generator.Graph_generator(network)
    print(gen.__dict__)
    gen.add_configfile("config_v6.csv")
    gen.generate_graph_from_config(4002)
    gen.graph.compute_dims()

def test_matcher(network="annette_bench1",config="config_v6.csv"):

    match = matcher.Graph_matcher(network,config)



def main():
    print("Main")
    network = "annette_bench1"
    #test_annette_to_model(network)
    #model = test_annette_to_model_from_config(network)
    test_matcher(network)

if __name__ == '__main__':
    main()
