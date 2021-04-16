import pytest
import sys
sys.path.append("./")
from pathlib import Path
import logging
import os

from annette import get_database
import annette.graphs as graphs

__author__ = "Matthias Wess"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"

def test_ONNXGraph_to_annette(network="cf_resnet50",inputs=None):
    network_file = get_database('graphs','onnx',network+'.onnx')
    onnx_network = graphs.ONNXGraph(network_file)
    annette_graph = onnx_network.onnx_to_annette(network, inputs)
    json_file = get_database( 'graphs', 'annette',
                     annette_graph.model_spec["name"]+'.json')
    annette_graph.to_json(json_file)

    assert True
    return 0 

def main():
    print("main")
    test_ONNXGraph_to_annette('squeezenet1.0-9',['data_0'])

if __name__ == '__main__':
    main()