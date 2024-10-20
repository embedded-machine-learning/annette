"""Annette Graph Utils

This Module contains the Graph Utilities to generate Annette readable graphs from MMDNN
or read directly from json.
"""
from __future__ import print_function

import json
import logging
import numpy as np
import sys
import os

import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.IR_graph import IRGraph, IRGraphNode, load_protobuf_from_file
from mmdnn.conversion.common.utils import *

from .annette_graph import AnnetteGraph

__author__ = "Matthias Wess"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache-2.0"


class MMGraph:
    """ MMDNN Graph Class

    Args:
        graphfile (str): MMDNN graphfile
        weightfile(str, optional): MMDNN weightfile, dropped anyways

    Attributes:
        IR_graph : mmdnn Intermediate Representation Graph 
    """

    def __init__(self, graphfile=None, weightfile=None):
        print("Initializing network...")

        self.graphfile = graphfile
        self.weightfile = weightfile

        self.IR_graph = IRGraph(self.graphfile)
        print("Loading weights...")
        self.IR_graph.build()
        self.IR_graph.model = 1
        print(self.IR_graph)
        
        if self.weightfile is None:
            logging.info("No weights file loaded\n")
        else:
            logging.info("Load weights...\n")
            try:
                self.weights_dict = np.load(
                    self.weightfile, allow_pickle=True).item()
            except:
                self.weights_dict = np.load(
                    self.weightfile, encoding='bytes', allow_pickle=True).item()

        self.analyze_net()
        print("Network analyzed successfully...\n")

    def analyze_net(self):
        """Walk through net and compute attributes"""

        # TODO look for DataInput layer and add if necessary
        """
        for layer in self.IR_graph.topological_sort:
            current_node = self.IR_graph.get_node(layer)
            node_type = current_node.type
            #find input layers            
            if not current_node.in_edges and not(current_node.type in ['DataInput']) :
                print(current_node.type)
        """

        for layer in self.IR_graph.topological_sort:
            current_node = self.IR_graph.get_node(layer)
            #node_type = current_node.type
            self.fix_shape_names(current_node)

    def fix_shape_names(self, layer):
        """Fixed shape_names

        Arguments:
            layer (obj): layer to fix names and shapes
        """
        if not(layer.type in ['yolo']):
            output_shape = layer.get_attr('_output_shape')
            # For tensorflow models it is called output_shapes
            if output_shape is None:
                output_shape = layer.get_attr('_output_shapes')
            output_shape = shape_to_list(output_shape[0])
            layer.set_attrs({'output_shape': output_shape})

        if not(layer.type in ['DataInput']):
            if(layer.in_edges):
                innode = self.IR_graph.get_node(layer.in_edges[0])
                input_shape = innode.get_attr('_output_shape')
                # For tensorflow models it is called output_shapes
                if input_shape is None:
                    input_shape = innode.get_attr('_output_shapes')
                input_shape = shape_to_list(input_shape[0])
                layer.set_attrs({'input_shape': input_shape})

    def fix_depthwise(self, layer):
        """Fixed depthwise layers 

        Arguments:
            layer (obj): layer to fix names and shapes
        """
        if layer.type in ['Conv']:
            output_shape = layer.get_attr('_output_shape')
            # For tensorflow models it is called output_shapes
            if output_shape is None:
                output_shape = layer.get_attr('_output_shapes')
            output_shape = shape_to_list(output_shape[0])
            group = layer.get_attr('group')
            if not (group is None):
                logging.debug(layer.name)
                logging.debug(group)
                logging.debug(output_shape)
                if group == output_shape[3]:
                    return 'DepthwiseConv'

        return layer.type

    def convert_to_annette(self, name):
        """Convert MMDNN to Annette graph

        Arguments:
            name (str): Network name 

        Return:
            annette_graph (obj)
        """
        annette_graph = AnnetteGraph(name)  # TODO

        for layer in self.IR_graph.topological_sort:
            current_node = self.IR_graph.get_node(layer)
            logging.debug(current_node.type)
            node_type = self.fix_depthwise(current_node)
            layer_dict = {'type': node_type}
            layer_name = current_node.name
            logging.debug(current_node.in_edges)
            logging.debug(current_node.out_edges)
            layer_dict['parents'] = current_node.in_edges
            layer_dict['children'] = current_node.out_edges

            attributes = ['output_shape', 'input_shape', 'kernel_shape',
                          'strides', 'pads', 'pooling_type', 'global_pooling', 'dilations','axis']

            for attr in attributes:
                tmp = current_node.get_attr(attr)
                if tmp is not None:
                    layer_dict[attr] = tmp
                    if layer_dict['type'] in ['DepthwiseConv'] and attr == 'kernel_shape':
                        tmp[3] = 1
                        layer_dict[attr] = tmp

            annette_graph.add_layer(layer_name, layer_dict)

        return annette_graph
