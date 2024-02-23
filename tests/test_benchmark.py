import pytest
import sys
sys.path.append("./")
from pathlib import Path
import logging
import os
import json

#logging.basicConfig(level=logging.DEBUG)

from annette.graph import MMGraph
from annette.graph import AnnetteGraph
import annette.benchmark.generator as generator
import annette.benchmark.matcher as matcher 

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
    gen.add_configfile("dummy.csv")
    gen.generate_graph_from_config(0)

    assert True

def test_compute_dims(network="annette_bench1"):
    gen = generator.Graph_generator(network)
    print(gen.__dict__)
    gen.add_configfile("config_v6.csv")
    gen.generate_graph_from_config(4002)
    gen.graph.compute_dims()


def test_matcher(network="annette_bench1",config="config_v6.csv"):

    match = matcher.Graph_matcher(network,config)


def test_annette_to_destruct(network="mobilenetv2-7-sim"):
    gen = generator.Graph_generator(network)
    gen.add_configfile("config_test.csv")
    #out = gen.generate_graph_from_config(0)
    #gen.generate_graph_from_config(0)
    rem_layers = []
    for j, l in enumerate(reversed(gen.init_graph.topological_sort)):
        out = gen.generate_graph_from_config(0)
        pb_path = get_database('graphs/destruct/'+str(network)+str(j)+'.pb')
        #gen.tf2_export_to_pb(out, save_path = get_database(pb_path))
        tflite_path = get_database('graphs/destruct/'+str(network)+str(j)+'.tflite')
        onnx_path = get_database('graphs/destruct/'+str(network)+str(j)+'.onnx')
        #gen.tf2_export_to_lite(['input'], out, {'input': [1,224,224,3]}, str(pb_path), str(tflite_path))
        #gen.lite_to_onnx(['input'], out, {'input': [1,224,224,3]}, str(tflite_path), str(onnx_path))
        #gen.tf_export_to_l(['input'], out, {'data': [1,160,80,3]}, str(pb_path), str(tflite_path))
        logging.debug("\n\n -----------next------------\n",gen.init_graph.model_spec['output_layers'])
        select = -1
        for i, l2 in enumerate(gen.init_graph.model_spec['output_layers']):
            logging.debug(i, l2)
            if len(gen.init_graph.model_spec['layers'][l2]['children']) == 0:
                select = i

        if select == -1:
            logging.error("All output layers have children. This should not be possible. Check your graph.")
            exit()
        name = gen.init_graph.model_spec['output_layers'][select]
        logging.debug("remove",name)
        rem_layers.append(gen.init_graph.model_spec['layers'][name])
        gen.init_graph.delete_layer(name)
    print(rem_layers)
   
    with open('layers.json', 'w') as outfile:
        json.dump(rem_layers, outfile) 
    assert True


def main():
    print("Main")
    network = "annette_bench_conv_max"
    #test_annette_to_model(network)
    model = test_annette_to_model_from_config(network)
    #test_matcher(network)
    #test_compute_dims(network)
    #test_annette_to_destruct('cf_reid')
    #test_annette_to_destruct()

if __name__ == '__main__':
    main()
