import annette.graph as graph
from annette import get_database
import onnx
from onnxsim import simplify
import os
import logging
from pathlib import Path
import pytest
import sys
sys.path.append("./")

__author__ = "Matthias Wess"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"

# set logging level to debug
#logging.basicConfig(level=logging.DEBUG)


def test_ONNXGraph_to_annette(network="cf_resnet50", inputs=None):
    network_file = get_database('graphs', 'onnx', network+'.onnx')
    onnx_model = onnx.load(network_file)
    simp = False
    if simp is True:
        simplified_file = get_database('graphs', 'onnx', network+'-sim.onnx')
        out_model, check = simplify(onnx_model, overwrite_input_shapes={'input':[1,3,640,640]})
        assert check, "Simplified ONNX model could not be validated"
    else:
        simplified_file = network_file
        out_model = onnx_model
    onnx.save(out_model, simplified_file)


    onnx_network = graph.ONNXGraph(network_file)
    annette_graph = onnx_network.onnx_to_annette(network, inputs)#, output_names=['/model.22/Concat_3'])
    #annette_graph = onnx_network.onnx_to_annette(network, inputs)
    json_file = get_database('graphs', 'annette',
                             annette_graph.model_spec["name"]+'.json')
    annette_graph.to_json(json_file)

    assert True
    return 0


def main():
    print("main")
    # test_ONNXGraph_to_annette('vgg16-7',['data'])
    test_ONNXGraph_to_annette('yolov8n-seg320-sim', [])
    #test_ONNXGraph_to_annette('yolov8n-sim', [])
    #test_ONNXGraph_to_annette('yolov8s-sim', [])
    #test_ONNXGraph_to_annette('yolov8m-sim', [])
    #test_ONNXGraph_to_annette('yolov8l-sim', [])
    #test_ONNXGraph_to_annette('yolov8x-sim', [])
    # test_ONNXGraph_to_annette('yolov5s',['images'])


if __name__ == '__main__':
    main()
