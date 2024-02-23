import logging
import os
import pickle
import pathlib
import sys
from annette.estimate import estimate
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from annette.utils import get_database
from pprint import pprint
from annette.graph.graph_util.annette_graph import AnnetteGraph
from annette.benchmark.generator import Graph_generator


def main():
    '''
    Run model estimation for a specified model on several input points given as 
    config CSV file.
    '''
    # NET_FOLDER = 'ties/micro-kernel/'
    NET_FOLDER = 'ties/multi-layer/'

    # NET_NAME = 'fully-connected'
    # NET_NAME = 'conv2d'
    # NET_NAME = 'avg-pool2d'
    # NET_NAME = 'dw-conv2d'
    # NET_NAME = 'conv-transpose2d'
    NET_NAME = 'annette_convnet'

    # CFG_NAME = 'aiware/compare_dwconv_regular.csv'
    # CFG_NAME = 'aiware/compare_conv-transpose2d_regular.csv'
    # CFG_NAME = 'aiware/compare_fc_regular.csv'
    # CFG_NAME = 'aiware/compare_conv_regular.csv'
    # CFG_NAME = 'aiware/compare_avgpool_regular.csv'
    CFG_NAME = 'aiware/convnet.csv'
    
    # CFG_NAME = 'aiware/test.csv'

    # RESULT_FILE_NAME = 'convfoo.p'
    # RESULT_FILE_NAME = '/local/steinmja/comparison/regular_data/conv_res_annette_mixed_test.p'
    # RESULT_FILE_NAME = '/local/steinmja/comparison/regular_data/conv_res_annette_stat_test.p'
    # RESULT_FILE_NAME = '/local/steinmja/comparison/regular_data/dwconv_res_annette_ref-roofline.p'
    # RESULT_FILE_NAME = '/local/steinmja/comparison/regular_data/convtranspose2d_res_annette_stat.p'
    RESULT_FILE_NAME = 'foobar.p'
    RESULT_FILE_NAME = None

    LAYER_MODEL = 'aiw-mixed'
    # MAPPING_MODEL = 'none'
    MAPPING_MODEL = sys.argv[1]

    # Set internal args needed for estimate function:
    args = ArgumentParser()
    args.network = f'{NET_NAME}_fixed'
    args.layer = LAYER_MODEL
    args.mapping = MAPPING_MODEL

    gen = Graph_generator(NET_FOLDER + NET_NAME, pad_pooling=False)
    gen.add_configfile(CFG_NAME)

    out_data = []

    for idx in [int(sys.argv[2])]:
    # for idx in range(600):
        gen.generate_graph_from_config(idx, framework='pytorch')
        gen.graph.to_json(get_database('graphs', 'annette', args.network + '.json'))

        total, layerwise = estimate(args)
        out_data.append(layerwise.loc[0]['time(ms)'])

    logging.debug(out_data)
    print(layerwise)
    print(total)

    if RESULT_FILE_NAME:
        with open(RESULT_FILE_NAME, 'wb') as ofile:
            pickle.dump(out_data, ofile)
            print(f'Dumped result pickle file at: {RESULT_FILE_NAME}')

    # Delete intermediate file(s):
    interm_file = get_database('graphs', 'annette', args.network + '.json')
    if interm_file.exists():
        os.remove(interm_file)


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    main()
