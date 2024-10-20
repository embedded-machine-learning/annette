import pytest
import sys
sys.path.append("./")
from pathlib import Path
import logging
import os

import graph_util

__author__ = "Matthias Wess"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"

logging.basicConfig(level=0)

def test_mmdnn_to_annette(network="cf_resnet50"):
    pb_file = Path('tests','data', network+'.pb')
    mmdnn_graph = graph_util.MMGraph(pb_file)
    annette_graph = mmdnn_graph.convert_to_annette(network)
    json_file = Path('tests','data', network+'.json')
    annette_graph.to_json(json_file)

    assert True

    return 

def main():
    print("main")
    network = "cf_inceptionv1"
    annette_graph = test_mmdnn_to_annette(network)

if __name__ == '__main__':
    main()
