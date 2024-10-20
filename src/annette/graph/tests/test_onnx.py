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

def test_ONNXGraph_to_annette(network="cf_resnet50",inputs=None):
    network_file = Path('tests','data',network+'.onnx')
    print("ONNXGraph_test")
    onnx_network = graph_util.ONNXGraph(network_file)
    annette_graph = onnx_network.onnx_to_annette(network, inputs)
    json_file = Path( 'tests', 'data', annette_graph.model_spec["name"]+'.json')
    annette_graph.to_json(json_file)

    assert True
    return 0 

def main():
    #test_ONNXGraph_to_annette('squeezenet1.0-9',['data_0'])
    test_ONNXGraph_to_annette('yolo_nas_s',[])


if __name__ == '__main__':
    main()