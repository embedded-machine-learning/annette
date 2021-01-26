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
from annette.utils import write_result

from annette import __version__

__author__ = "mwessley"
__copyright__ = "mwessley"
__license__ = "boost"

_logger = logging.getLogger(__name__)

def estimate(args):
    """Estimate example function

    Args:
      network name (str): network name 

    Returns:
      float: estimated time in ms 
    """
    model = AnnetteGraph(args.network, Path('database','graphs','annette',args.network+'.json'))

    if args.mapping != "none":
      opt = Mapping_model.from_json(Path('database','models','mapping',args.mapping+'.json'))
      #opt = Mapping_model.from_json("tests/test_data/mapping/ov2.json")
      opt.run_optimization(model)

    # LOAD MODELS
    mod = Layer_model.from_json(Path('database','models','layer',args.layer+'.json'))
        
    # APPLY ESTIMATION
    res = mod.estimate_model(model)
    write_result(args.network, res, args.mapping, args.layer, Path('database','results'))

    return res[0], res[2] 

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
        dest="mapping",
        help="mapping model file",
        type=str,
        default="none",
        metavar="STRING")
    parser.add_argument(
        dest="layer",
        help="layer model file",
        default="none",
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
    print(args.network)
    total, layerwise  = estimate(args)
    print("The network {} is layer results are: \n{}".format(args.network, layerwise))
    print("The network {} is executed in {} ms ".format(args.network, total))

def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])

if __name__ == "__main__":
    run()
