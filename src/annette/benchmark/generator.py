from __future__ import print_function
from annette.estimation import layers
import json
import numpy as np
import pandas as pd
import pickle as pkl
import logging

from annette import get_database 

def generate_tf2_model(graph):
    """generates Tensorflow 2 graph out of ANNETTE graph description
    and stores to benchmark/graphs/tf2/ directory

    Args:
        graph :obj:`annette.graph.AnnetteGraph`: annette graph description to generate the tf2 graph from
    """

    # generate tensorflow model and export to out_file

    # with __dict__ we can see the content of the class
    logging.debug(graph.__dict__)

    # model_spec contains some info about the model
    for key, value  in graph.model_spec.items():
        logging.debug(key)
        logging.debug(value)

    network_name = graph.model_spec['name']

    filename = get_database( 'benchmark', 'graphs' ,'tf2', network_name+'.pb')
    logging.debug("Stored to: %s" % filename)
    
def generate_graph_from_config(config_df, graph, specific):

    # get config dataframe and generate graph

    # can be used as to generate input for generate_tf2_model

    # return annette graph
    return None


