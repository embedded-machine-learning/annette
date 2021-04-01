import pytest
import sys
sys.path.append("./")
from pathlib import Path
import logging
import os

logging.basicConfig(level=logging.DEBUG)

from annette.graph import MMGraph
from annette.graph import AnnetteGraph
import annette.benchmark.generator

__author__ = "Matthias Wess"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"


def test_annette_to_model(network="cf_reid"):
    json_file = Path('database','graphs','annette',network+'.json')
    annette_graph = AnnetteGraph(network, json_file)

    # execute the function under test
    annette.benchmark.generator.generate_tf2_model(annette_graph)
    assert True


def main():
    print("Main")
    network = "cf_reid"
    test_annette_to_model(network)

if __name__ == '__main__':
    main()
