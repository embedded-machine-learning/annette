import pytest
import sys
sys.path.append("./")
from pathlib import Path
import logging
import os

from annette.graph import AnnetteGraph
from annette.utils import get_database
__author__ = "Matthias Wess"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"

def test_annette_from_json(network="cf_resnet50"):
    json_file = get_database('graphs','annette',network+'.json')
    annette_graph = AnnetteGraph(network, json_file)

    assert True

def test_resize_annette(network="squeezenet1_0"):
    json_file = get_database('graphs','annette',network+'.json')
    annette_graph = AnnetteGraph(network, json_file)
    annette_graph.scale_input_resolution(0.5)
    print(annette_graph)

    assert True

#TODO: test fuse Layers -> test delete Layers

def main():
    network = "squeezenet1_0"
    #test_annette_from_json(network=network)
    test_resize_annette(network=network)

if __name__ == '__main__':
    main()
