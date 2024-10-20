"""Annette Graph Utils

This Module contains the Graph Utilities to generate Annette readable graphs from MMDNN
or read directly from json.
"""
from __future__ import print_function

import json
import logging
import numpy as np
import sys
from copy import copy, deepcopy
from functools import reduce

from .annette_graph import AnnetteGraph

__author__ = "Matthias Wess"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache-2.0"

class NNMeterGraph():
    """NNMeter DNN Graph Description object.

    Args:
        name (str): network name
        json_file (str, optional): location of the .json file the network
            will be generated from. Alternatively an empty network graph will
            be generated

    Attributes:
        :var model_spec (dict): model_spec dictionary with the following
        model_spec[name] (str): network name
        model_spec[layers] (dict): mmdnn style description of the layers
        model_spec[input_layers] (list): list of the input layer names
        model_spec[output_layers] (list): list of the output layer names
    """

    def __init__(self, name, json_file=None, in_dict=None):
        if in_dict is None:
            self.model_spec = {}
            self.model_spec['name'] = name
            self.model_spec['layers'] = {}
            self.model_spec['input_layers'] = []
            self.model_spec['output_layers'] = []
        else:
            self.model_spec = deepcopy(in_dict)
            self.model_spec['layers'] = deepcopy(in_dict['graph'])
            self.model_spec['input_layers'] = []
            self.model_spec['output_layers'] = []

        if json_file:
            print("Loading from file: ", json_file)
            with open(json_file, 'r') as f:
                self.model_spec = json.load(f)
                self.model_spec['layers'] = deepcopy(self.model_spec['graph'])
                
        self.model_spec['input_layers'] = self._find_input_layers()
        self.model_spec['output_layers'] = self._find_output_layers()
        self.topological_sort = self._get_topological_sort()
        
    def _find_input_layers(self):
        """Find input layers

        Returns:
            input_layers (list)
        """
        input_layers = []
        #print("Finding input layers")
        #print(self.model_spec['layers'])
        for key, layer in self.model_spec['layers'].items():
            self.model_spec['layers'][key].update(deepcopy(layer['attr']))
            self.model_spec['layers'][key].update(deepcopy(layer['attr']))
            self.model_spec['layers'][key]['parents'] = deepcopy(layer['inbounds'])
            
            
            #if len(self.model_spec['layers'][key]['output_shape']) > 0:
            #    self.model_spec['layers'][key]['output_shape'] = self.model_spec['layers'][key]['output_shape'][0]
            #if len(self.model_spec['layers'][key]['input_shape']) > 0:
            #    self.model_spec['layers'][key]['input_shape'] = self.model_spec['layers'][key]['input_shape'][0]
            
            if len(self.model_spec['layers'][key]['parents']) == 0:
                input_layers.append(key)
            
            else:
                for n, p in enumerate(self.model_spec['layers'][key]['parents']):
                    if p not in self.model_spec['layers'].keys():
                        #print(p, "not in layers")
                        #print(self.model_spec['layers'][key]['parents'])
                        #print(self.model_spec['layers'][key]['parents'][n])
                        del self.model_spec['layers'][key]['parents'][n]
                        #print(self.model_spec['layers'][key]['parents'])
            #print(self.model_spec['layers'][key])
        return input_layers

    def _find_output_layers(self):
        """Find output layers

        Returns:
            output (list)
        """
        output_layers = []
        for key in self.model_spec['layers'].keys():
            self.model_spec['layers'][key]['children'] = self.model_spec['layers'][key]['outbounds']
            #print(self.model_spec['layers'][key]['children'])
            if len(self.model_spec['layers'][key]['children']) == 0:
                output_layers.append(key)
        return output_layers

    def _get_topological_sort(self):
        """Resort Graph

        Returns:
            Topological Sort
        """
        self.topological_sort = self.model_spec['input_layers'][:]
        idx = 0
        for n in self.model_spec['layers']:
            self.model_spec['layers'][n]['left_parents'] = len(
                self.model_spec['layers'][n]['parents'])
        while idx < len(self.topological_sort):
            name = self.topological_sort[idx]
            current_node = self.model_spec['layers'][name]
            for next_node in current_node['children']:
                next_node_info = self.model_spec['layers'][next_node]
                # one node may connect another node by more than one edge.
                self.model_spec['layers'][next_node]['left_parents'] -= \
                    self._check_left_parents(name, next_node_info)
                if next_node_info['left_parents'] == 0:
                    self.topological_sort.append(next_node)
            idx += 1
        logging.debug(self.topological_sort)
        return self.topological_sort

    def _check_left_parents(self, in_node_name, node):
        count = 0
        for in_edge in node['parents']:
            if in_node_name == in_edge.split(':')[0]:
                count += 1
        return count

    def to_json(self, filename):
        with open(filename, 'w') as f:
            f.write(json.dumps(self.model_spec, indent=4))
        print("Stored to %s" % filename)


    def convert_to_annette(self, name):
        """Convert MMDNN to Annette graph

        Arguments:
            name (str): Network name 

        Return:
            annette_graph (obj)
        """
        annette_graph = AnnetteGraph(name)  # TODO
        
        for layer in self.topological_sort:
            current_node = self.model_spec['layers'][layer]
            logging.debug(current_node['type'])
            layer_dict = {'type': current_node['type'],}
            layer_name = current_node['name']
            logging.debug(current_node['parents'])
            logging.debug(current_node['children'])
            layer_dict['parents'] = current_node['parents']
            layer_dict['children'] = current_node['children']

            attributes = ['kernel_shape',
                          'strides', 'pads', 'pooling_type', 'global_pooling', 'dilations','axis']

            if len(current_node['output_shape']) > 0:
                layer_dict['output_shape'] = current_node['output_shape'][0]
            if len(current_node['input_shape']) > 0:
                layer_dict['input_shape'] = current_node['input_shape'][0]
            if layer_dict['type'] in ['MatMul']:
                layer_dict['type'] = 'FullyConnected'
            #print(current_node)
            for attr in attributes:
                if attr in current_node:
                    tmp = current_node[attr]
                    if tmp is not None:
                        layer_dict[attr] = tmp
                        if layer_dict['type'] in ['DepthwiseConv'] and attr == 'kernel_shape':
                            tmp[3] = 1
                            layer_dict[attr] = tmp
                        if layer_dict['type'] in ['DepthwiseConv2dNative'] and attr == 'kernel_shape':
                            layer_dict['type'] = 'DepthwiseConv'
                            layer_dict[attr].append(layer_dict['input_shape'][3])
                            layer_dict[attr].append(1)
                        if layer_dict['type'] in ['Conv2D'] and attr == 'kernel_shape':
                            layer_dict['type'] = 'Conv'
                            layer_dict[attr].append(layer_dict['input_shape'][3])
                            layer_dict[attr].append(layer_dict['output_shape'][3])

            annette_graph.add_layer(layer_name, layer_dict)
            annette_graph.model_spec['ncs2_latency'] = self.model_spec['myriadvpu_openvino2019r2']
        #annette_graph.model_spec

        return annette_graph
