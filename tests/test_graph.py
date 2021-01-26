import pytest
import sys
sys.path.append("./")
from annette.graph import MMGraph
from annette.graph import AnnetteGraph
from pathlib import Path
import os
import logging

__author__ = "mwessley"
__copyright__ = "mwessley"
__license__ = "mit"


def test_MMGraph_to_annette(network="cf_resnet50"):
    print("MMGraph_test")
    graphfile = Path('tests','test_data','graph_mmdnn',network+'.pb')
    weightfile = None
    mmdnn_graph = MMGraph(graphfile, weightfile)
    annette_graph = mmdnn_graph.convert_to_annette(network)
    json_file = "tests/test_data/graph_annette/"+annette_graph.model_spec["name"]+".json"
    annette_graph.to_json(json_file)

    assert True
    return annette_graph

def test_annette_from_json(network="cf_resnet50"):
    json_file = "tests/test_data/graph_annette/"+network+".json"
    new_annette_graph = AnnetteGraph(network, json_file)

    logging.warning("test")
    assert True

#TODO: test fuse Layers -> test delet Layers
#TODO: tensorflow weird naming test


def convert_nas():
    network = "model_224_0"
    netlist = os.listdir("tests/test_data/nas/nas_mmdnn/")
    netlist = [n.split('.json')[0] for n in netlist if n.endswith(".json")]
    print(netlist)
    for network in netlist:
        annette_graph = test_MMGraph_to_annette(network)
        print("annette:\n")
        print(annette_graph.model_spec)
        annette_graph.to_json("tests/test_data/nas/nas_annette/"+annette_graph.model_spec["name"]+".json")
        print("GO!!!\n\n")

def main():
    print("main")
    network = "Resnet_original"
    annette_graph = test_MMGraph_to_annette(network)
    print("annette:\n")
    print(annette_graph.model_spec)
    annette_graph.to_json("tests/test_data/graph_annette/"+annette_graph.model_spec["name"]+".json")
    print("GO!!!\n\n")

if __name__ == '__main__':
    main()
