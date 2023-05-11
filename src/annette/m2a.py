# -*- coding: utf-8 -*-
import argparse
import logging
import os
import sys
from pathlib import Path

from annette import __version__
from annette.estimation.layer_model import Layer_model
from annette.estimation.mapping_model import Mapping_model
import annette.graph as graph
from annette import get_database 

sys.path.append("./")


__author__ = "Matthias Wess"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"

_logger = logging.getLogger(__name__)


def mmdnn_to_annette(args):
    """Convert MMDNN graph .pb-File to annette json format and stores to Annette graph std path.

    Args:
      network name (str): network name 

    Returns:
      :obj:`annette.AnnetteGraph`: AnnetteGraph object
    """
    graphfile = get_database( 'graphs', 'mmdnn', args.network+'.pb')
    if(os.path.exists(graphfile)):
        print("Graph found")
    elif(os.path.exists(args.network)):
        print("Graph-file detected")
        # extract network name
        args.network = os.path.split(args.network)[1].split('.pb')[0]
        graphfile = get_database( 'graphs', 'mmdnn', args.network+'.pb')
    else:
        logging.error("File not found")

    weightfile = None
    mmdnn_graph = graph.MMGraph(graphfile, weightfile)
    annette_graph = mmdnn_graph.convert_to_annette(args.network)
    json_file = get_database( 'graphs', 'annette',
                     annette_graph.model_spec["name"]+'.json')
    annette_graph.to_json(json_file)

    return annette_graph

def onnx_to_annette(args):
    """Convert ONNNX graph .onnx-File to annette json format and stores to Annette graph std path.

    Args:
      network name (str): network name 
      input node name (str): input node name 

    Returns:
      :obj:`annette.AnnetteGraph`: AnnetteGraph object
    """

    graphfile = get_database('graphs','onnx',args.network+'.onnx')
    if(os.path.exists(graphfile)):
        print("Graph found")
    elif(os.path.exists(args.network)):
        print("Graph-file detected")
        # extract network name
        args.network = os.path.split(args.network)[1].split('.onnx')[0]
        graphfile = get_database( 'graphs', 'onnx', args.network+'.onnx')
    else:
        logging.error("File not found")
    onnx_network = graph.ONNXGraph(graphfile)
    annette_graph = onnx_network.onnx_to_annette(args.network, args.inputs)
    if annette_graph == None:
        logging.error("Input node not found")
        return None
    json_file = get_database( 'graphs', 'annette',
                     annette_graph.model_spec["name"]+'.json')
    annette_graph.to_json(json_file)

    return annette_graph 


def parse_args(args):
    """Parse command line parameters.

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="ANNETTE - Accurate Neural Network Execution Time Estimation")
    parser.add_argument(
        "--version",
        action="version",
        version="annette {ver}".format(ver=__version__))
    parser.add_argument(
        "-n",
        "--network",
        dest="network",
        help="network name to estimate",
        type=str,
        metavar="net")
    parser.add_argument(
        "-i",
        "--inputs",
        dest="inputs",
        default=None,
        help="input_nodes in list form e.g. ['data']",
        type=str,
        metavar="in")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO)
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG)
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging.

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def main(args, iformat='mmdnn'):
    """Main entry point allowing external calls.

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    if iformat == 'mmdnn':
        mmdnn_to_annette(args)
    elif iformat == 'onnx':
        onnx_to_annette(args)
    else:
        logging.info('Unknown input format')


def run():
    """Entry point for console_scripts."""
    main(sys.argv[1:])

def run_onnx():
    """Entry point for console_scripts."""
    main(sys.argv[1:], iformat='onnx')

if __name__ == "__main__":
    run()
