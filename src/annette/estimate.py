# -*- coding: utf-8 -*-
import argparse
import logging
import os
import sys
from pathlib import Path

from annette import __version__
from annette.estimation.layer_model import Layer_model
from annette.estimation.mapping_model import Mapping_model
from annette.graph import AnnetteGraph
from annette.utils import write_result
from annette import get_database 

sys.path.append("./")

__author__ = "Matthias Wess"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"

_logger = logging.getLogger(__name__)

def estimate(args):
    """Estimate example function

    Args:
      network name (str): network name 

    Returns:
      float: estimated time in ms 
    """
    model = AnnetteGraph(args.network, get_database('graphs', 'annette', args.network+'.json'))

    if args.mapping != "none":
        opt = Mapping_model.from_json(
            get_database('models', 'mapping', args.mapping+'.json'))
        opt.run_optimization(model)

    # LOAD MODELS
    mod = Layer_model.from_json(
        get_database('models', 'layer', args.layer+'.json'))

    # APPLY ESTIMATION
    res = mod.estimate_model(model)
    write_result(args.network, res, args.mapping,
                 args.layer, get_database('results'))

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
        metavar="network")
    parser.add_argument(
        dest="mapping",
        help="mapping model file",
        type=str,
        default="none",
        metavar="mapping_model")
    parser.add_argument(
        dest="layer",
        help="layer model file",
        default="none",
        type=str,
        metavar="layer_model")
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

    print(args)
    args = parse_args(args)
    print(args.__dict__)
    setup_logging(args.loglevel)
    print(args.network)
    total, layerwise = estimate(args)
    print("The network {} is layer results are: \n{}".format(
        args.network, layerwise))
    print("The network {} is executed in {} ms ".format(args.network, total))

    return total, layerwise


def run():
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
