import pytest
import sys
sys.path.append("./")
from pathlib import Path
import logging
import os

from annette.graph import MMGraph
from annette.graph import AnnetteGraph

__author__ = "Matthias Wess"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"

def test_MMGraph_to_annette(network="cf_resnet50"):

    print("MMGraph_test")
    graphfile = Path('database','graphs','mmdnn',network+'.pb')
    weightfile = None
    mmdnn_graph = graphs.MMGraph(graphfile, weightfile)
    annette_graph = mmdnn_graph.convert_to_annette(network)
    json_file = Path('database','graphs','annette',annette_graph.model_spec["name"]+'.json')
    annette_graph.to_json(json_file)

    assert True
    return annette_graph

def test_annette_from_json(network="cf_resnet50"):
    json_file = Path('database','graphs','annette',network+'.json')
    new_annette_graph = graphs.AnnetteGraph(network, json_file)

    logging.warning("test")
    assert True

#TODO: test fuse Layers -> test delete Layers
#TODO: tensorflow weird naming test

def main():
    print("main")
    network = "Resnet_original"
    annette_graph = test_MMGraph_to_annette(network)
    print("annette:\n")
    print(annette_graph.model_spec)
    json_file = Path('database','graphs','annette',annette_graph.model_spec["name"]+'.json')
    annette_graph.to_json(json_file)

if __name__ == '__main__':
    main()
