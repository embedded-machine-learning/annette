# -*- coding: utf-8 -*-
import argparse
import sys
import logging
import os
from pathlib import Path
sys.path.append("./")

from annette.graph import MMGraph
from annette.graph import AnnetteGraph
from annette.estimation.layer_model import Layer_model 
from annette.estimation.mapping_model import Mapping_model 

from annette import __version__

__author__ = "mwessley"
__copyright__ = "mwessley"
__license__ = "boost"

_logger = logging.getLogger(__name__)

def mmdnn_to_annette(args):
    """Convert MMDNN graph .pb-File to annette json format and stores to Annette graph std path

    Args:
      network name (str): network name 

    Returns:
      :obj:`annette.AnnetteGraph`: AnnetteGraph object
    """
    graphfile = Path('database','graphs','mmdnn',args.network+'.pb')
    if(os.path.exists(graphfile)):
      print("Graph found")
    elif(os.path.exists(args.network)):
      print("Graph-file detected")
      # extract network name
      args.network = os.path.split(args.network)[1].split('.pb')[0]
      graphfile = Path('database','graphs','mmdnn',args.network+'.pb')
    else:
      logging.error("File not found")

    weightfile = None
    mmdnn_graph = MMGraph(graphfile, weightfile)
    annette_graph = mmdnn_graph.convert_to_annette(args.network)
    json_file = Path('database','graphs','annette',annette_graph.model_spec["name"]+'.json')
    annette_graph.to_json(json_file)

    return annette_graph 


def parse_args(args):
    """Parse command line parameters

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
        dest="network",
        help="network name to estimate",
        type=str,
        metavar="STRING")
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
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")

def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    mmdnn_to_annette(args)

def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])

if __name__ == "__main__":
    run()
