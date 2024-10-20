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
import texttable

from collections import OrderedDict

def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

__author__ = "Matthias Wess"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache-2.0"


class AnnetteGraph():
    """Annette DNN Graph Description object.

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
        self.model_spec = dict()
        self.model_spec['name'] = name
        self.model_spec['layers'] = dict()

        if json_file:
            print("Loading from file: ", json_file)
            with open(json_file, 'r') as f:
                self.model_spec = json.load(f)
        if not 'input_layers' in self.model_spec:
            self.model_spec['input_layers'] = []
        if not 'output_layers' in self.model_spec:
            self.model_spec['output_layers'] = []
        self._make_input_layers()
        self._make_output_layers()
        self.topological_sort = self._get_topological_sort()
    
    def scale_input_resolution(self, factor=1.):
        # loop over all input layers
        for layer in self.model_spec['input_layers']:
            print("Scaling input resolution of layer: ", layer)
            # get input shape
            print(self.model_spec['layers'][layer])
            output_shape = self.model_spec['layers'][layer]['output_shape']
            # scale input shape
            for i in (1,2):
                output_shape[i] = int(output_shape[i] * factor)
            # set input shape
            self.model_spec['layers'][layer]['output_shape'] = output_shape
        # recompute dims
        self._make_input_layers()
        self._make_output_layers()
        self.topological_sort = self._get_topological_sort()
        self.compute_dims()
        return True
            
    def resort(self):
        """Resort Layers """
        self._make_input_layers()
        self._make_output_layers()
        self.topological_sort = self._get_topological_sort()
        return True

    def print_as_table(self):
        table = texttable.Texttable(0)
        table.set_cols_align(['l', 'l', 'l', 'l'])
        table.header(['Layer Name', 'Input Shape', 'Kernel Shape', 'Output Shape'])

        for l in self.topological_sort:
            inshape = outshape = kshape = '--'
            params = self.model_spec['layers'][l]
            if 'input_shape' in params.keys():
                inshape = params['input_shape']
            if 'output_shape' in params.keys():
                outshape = params['output_shape']
            outshape = params['output_shape']
            if 'kernel_shape' in params.keys():
                kshape = params['kernel_shape']
            table.add_row([l, inshape, kshape, outshape])

        print()
        print('AnnetteGraph information (topologically sorted):')
        print(table.draw())
        print()

    def add_layer(self, layer_name, layer_attr, resort = False):
        """Add a layer to the network description graph -> model_spec[layers]

        Args:
            layer_name (str): Name of the Layer to add
            layer_attr (dict): mmdnn style layer attributes

        Returns:
            True if successful, False otherwise.

        TODO:
            * check for existing parents
            * check for duplicate layer_names
            * check for valid attributes
        """
        self.model_spec['layers'][layer_name] = layer_attr.copy()
        if resort:
            for p in layer_attr['parents']:
                if p in self.model_spec['layers'].keys():
                    self.model_spec['layers'][p]['children'].append(layer_name)
            self._make_input_layers()
            self._make_output_layers()
            self.topological_sort = self._get_topological_sort()
        return True

    def _make_output_layers(self):
        self.model_spec['output_layers'] = []
        for name, layer in self.model_spec['layers'].items():
            if len(layer['children']) == 0:
                self.model_spec['output_layers'].append(name)

    def _make_input_layers(self, rebuild=False):
        self.model_spec['input_layers'] = []
        for name, layer in self.model_spec['layers'].items():
            self.model_spec['layers'][name]['left_parents'] = len(
                layer['parents'])
            if len(layer['parents']) == 0:
                self.model_spec['input_layers'].append(name)

    def fuse_layer(self, primary, secondary):
        """Fuse secondary to primary layer

        Args:
            primary (str): Name of the primary layer
            secondary (str): Name of the secondary layer
        Returns:
            True if successful, False otherwise.
        """
        if secondary in self.model_spec['layers'] and primary in self.model_spec['layers']:
            logging.debug("fuse layer %s to layer %s" % (secondary, primary))
            self.model_spec['layers'][primary][self.model_spec['layers'][secondary]['type']] = \
                self.model_spec['layers'][secondary]
            self.delete_layer(secondary)
            return True
        else:
            return False

    def split_layer(self, primary, secondary):
        """Split primary to seconday layers

        Args:
            primary (str): Name of the primary layer
            secondary (list): Names of the secondary layers
        Returns:
            True if successful, False otherwise.
        """
        if primary in self.model_spec['layers']:
            logging.debug("Split layer %s" % (primary))
            logging.debug(self.model_spec['layers'][primary])
            logging.debug(secondary)
            new_layers = []
            for n, in_layer in enumerate(secondary):
                new_name = primary+"_"+in_layer
                new_layers.append(new_name)
                logging.debug(n, in_layer)
                logging.debug("Add layer")
                if n == 0:
                    # change layer name
                    self.model_spec['layers'][new_name] = self.model_spec['layers'].pop(
                        primary)
                else:
                    self.add_layer(
                        new_name, self.model_spec['layers'][new_layers[0]])
                # change layer type
                self.model_spec['layers'][new_name]['type'] = in_layer
                logging.debug(self.model_spec['layers'][new_name])
            logging.debug(new_layers)
            for n, in_layer in enumerate(new_layers):
                logging.debug("Edit %s" % in_layer)
                logging.debug(self.model_spec['layers'][in_layer])
                if n == 0:
                    # change parents
                    for p in self.model_spec['layers'][in_layer]['parents']:
                        logging.debug("parent: %s" %
                                      self.model_spec['layers'][p]['children'])
                        self.model_spec['layers'][p]['children'] = \
                            [in_layer if x == primary else x for x in self.model_spec['layers'][p]['children']]
                        logging.debug("new_parent: %s" %
                                      self.model_spec['layers'][p]['children'])
                    logging.debug("new layer name: %s" % in_layer)
                    # change children
                    logging.debug("new_child:" + new_layers[n+1])
                    self.model_spec['layers'][in_layer]['children'] = [
                        new_layers[n+1]]
                elif n == len(new_layers)-1:
                    logging.debug(n)
                    # change parents
                    logging.debug("new_parent:" + new_layers[n-1])
                    self.model_spec['layers'][in_layer]['parents'] = [
                        new_layers[n-1]]
                    # change children
                    for c in self.model_spec['layers'][in_layer]['children']:
                        logging.debug(c)
                        logging.debug("children: %s" %
                                      self.model_spec['layers'][c]['parents'])
                        self.model_spec['layers'][c]['parents'] = \
                            [in_layer if x ==
                                primary else x for x in self.model_spec['layers'][c]['parents']]
                        logging.debug("children: %s" %
                                      self.model_spec['layers'][c]['parents'])
                    logging.debug("current layer name: %s" % new_name)
                else:
                    self.model_spec['layers'][in_layer]['parents'] = [
                        new_layers[n-1]]
                    self.model_spec['layers'][in_layer]['children'] = [
                        new_layers[n+1]]
            return True

    def delete_layer(self, name):
        """Delete layer from Graph

        Args:
            name (str): Name of the layer to delete
        Returns:
            True if successful, False otherwise.
        """
        if name in self.model_spec['layers']:
            logging.info("Deleting layer %s" % name)

            if len(self.model_spec['layers'][name]['parents']) > 0:
                parents = (self.model_spec['layers'][name]['parents'])
            else:
                parents = []
                logging.debug("No parents")

            if len(self.model_spec['layers'][name]['children']) > 0:
                children = (self.model_spec['layers'][name]['children'])
            else:
                children = []
                logging.debug("No children")

            for p in parents:
                logging.debug("parents: %s" % p)
                logging.debug(self.model_spec['layers'][p]['children'])
                self.model_spec['layers'][p]['children'].remove(name)
                for c in children:
                    self.model_spec['layers'][p]['children'].append(c)
                logging.debug(self.model_spec['layers'][p]['children'])

            for c in children:
                logging.debug("children: %s" % c)
                logging.debug(self.model_spec['layers'][c]['parents'])
                self.model_spec['layers'][c]['parents'].remove(name)
                for p in parents:
                    self.model_spec['layers'][c]['parents'].append(p)
                logging.debug(self.model_spec['layers'][c]['parents'])

            if name in self.model_spec['input_layers']:
                for c in children:
                    if c not in self.model_spec['input_layers']:
                        self.model_spec['input_layers'].append(c)
                self.model_spec['input_layers'].remove(name)

            if name in self.model_spec['output_layers']:
                for p in parents:
                    if p not in self.model_spec['output_layers']:
                        self.model_spec['output_layers'].append(p)
                self.model_spec['output_layers'].remove(name)

            del self.model_spec['layers'][name]
            self._get_topological_sort()
            return self
        else:
            logging.warning("Layer %s does not exists" % name)
            return False
    
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

    def compute_dims(self):
        #loop through layers
        logging.debug(self.model_spec)
        for l_name in self._get_topological_sort():
            logging.debug("current layer")
            l_attr = self.model_spec['layers'][l_name]
            logging.debug(l_name)
            logging.debug(l_attr)

            l_type = l_attr['type'] 
            #check if input layer
            if l_name not in self.model_spec['input_layers']: 

                #Compute Output Size
                if hasattr(self, "compute_dims_" + l_type):
                    func = getattr(self, "compute_dims_" + l_type)
                else:
                    func = getattr(self, "compute_dims_base")
                self.model_spec['layers'][l_name] = func(l_name)

            logging.debug("changed attributes to:")
            logging.debug(l_name)
            logging.debug(l_attr)

    def compute_dims_Concat(self, l_name):
        p_name = self.model_spec['layers'][l_name]['parents']
        l_attr = self.model_spec['layers'][l_name]
        l_type = l_attr['type'] 
        #get input from parents
        if 'axis' in l_attr:
            axis = l_attr['axis']
        else:
            #compare input and output shape and select axis where they differ
            axis = 0
            for n in range(len(l_attr['input_shape'])):
                if l_attr['output_shape'][n] != l_attr['input_shape'][n]:
                    axis = n
                    break
        l_attr['axis'] = axis

        tmp_sum = 0
        l_attr['input_shape'] = [0]*len(p_name)
        for n, p_tmp in enumerate(p_name):
            p_attr = self.model_spec['layers'][p_tmp]
            tmp_sum = tmp_sum + p_attr['output_shape'][l_attr['axis']]
            l_attr['input_shape'][n] = copy(p_attr['output_shape'])
            l_attr['output_shape'] = copy(p_attr['output_shape'])
        l_attr['output_shape'][l_attr['axis']] = tmp_sum
        logging.debug(tmp_sum)
        return deepcopy(l_attr)


    def compute_dims_Pool(self, l_name):
        l_attr = self.model_spec['layers'][l_name]
        l_type = l_attr['type'] 
        p_name = self.model_spec['layers'][l_name]['parents']
        #if inputshape is like kernel shape assume global pooling and adapt kernel size
        global_average = False
        if l_attr['input_shape'][1:3] == l_attr['kernel_shape'][1:3]:
            global_average = True
            
        #get input from parents
        if len(p_name) == 1:
            p_attr = self.model_spec['layers'][p_name[0]]
            l_attr['input_shape'] = p_attr['output_shape']
            if global_average:
                l_attr['kernel_shape'][1:3] = l_attr['input_shape'][1:3]
        else:
            raise NotImplementedError
        if 'strides' in l_attr:
            #add padding to input shape
            tmp_pads = [0]*(len(l_attr['pads'])//2)
            for n in range(len(l_attr['pads'])//2):
                tmp_pads[n] = l_attr['pads'][n]+l_attr['pads'][n+len(tmp_pads)]
            tmp_inputs = l_attr['input_shape']
            print(tmp_inputs, tmp_pads)
            tmp_input_shape = [x+y for x, y in zip(tmp_inputs, tmp_pads)]
            print(tmp_input_shape)
            l_attr['output_shape'] = [int((x-(y-1))/z) for x, y, z in zip(tmp_input_shape, l_attr['kernel_shape'], l_attr['strides'])]
            print(l_attr['output_shape'])
        elif reduce(lambda x,y: x*y, l_attr['pads']) == 0:
            l_attr['output_shape'] = [x-int((y-1)) for x, y in zip(l_attr['input_shape'], l_attr['kernel_shape'])]
            print(l_attr['output_shape'])
            exit()
        else:
            l_attr['output_shape'] = l_attr['input_shape']
        l_attr['output_shape'][-1] = l_attr['input_shape'][-1]
        logging.debug(l_attr)
        return deepcopy(l_attr)

    def compute_dims_Flatten(self, l_name):
        p_name = self.model_spec['layers'][l_name]['parents']
        l_attr = self.model_spec['layers'][l_name]
        l_type = l_attr['type'] 

        p_attr = self.model_spec['layers'][p_name[0]]
        l_attr['input_shape'] = p_attr['output_shape']
        size = reduce(lambda x, y: x*y, p_attr['output_shape'][1:])
        l_attr['output_shape'] = [l_attr['input_shape'][0], size]
            
        logging.debug(l_attr)
        return deepcopy(l_attr)

    def compute_dims_Reshape(self, l_name):
        p_name = self.model_spec['layers'][l_name]['parents']
        l_attr = self.model_spec['layers'][l_name]
        l_type = l_attr['type'] 

        p_attr = self.model_spec['layers'][p_name[0]]
        l_attr['input_shape'] = p_attr['output_shape']
            
        logging.debug(l_attr)
        return deepcopy(l_attr)

    def compute_dims_FullyConnected(self, l_name):
        l_attr = self.model_spec['layers'][l_name]
        l_type = l_attr['type'] 
        p_name = self.model_spec['layers'][l_name]['parents']
        #get input from parents
        if len(p_name) == 1:
            p_attr = self.model_spec['layers'][p_name[0]]
            l_attr['input_shape'] = p_attr['output_shape']
        else:
            raise NotImplementedError

        l_attr['output_shape'][0] = l_attr['input_shape'][0]
        logging.debug(l_attr)
        return deepcopy(l_attr)

    def compute_dims_MatMul(self, l_name):
        l_attr = self.model_spec['layers'][l_name]
        l_type = l_attr['type'] 
        p_name = self.model_spec['layers'][l_name]['parents']
        #get input from parents
        if len(p_name) == 1:
            p_attr = self.model_spec['layers'][p_name[0]]
            l_attr['input_shape'][0] = p_attr['output_shape'][0]
        else:
            raise NotImplementedError

        l_attr['output_shape'][0] = l_attr['input_shape'][0]
        logging.debug(l_attr)
        return deepcopy(l_attr)

    def compute_dims_Conv(self, l_name):
        #TODO: add dilation
        l_attr = self.model_spec['layers'][l_name]
        l_type = l_attr['type'] 
        p_name = self.model_spec['layers'][l_name]['parents']
        #get input from parents
        if len(p_name) == 1:
            p_attr = self.model_spec['layers'][p_name[0]]
            l_attr['input_shape'] = p_attr['output_shape']
        else:
            raise NotImplementedError
        if 'strides' in l_attr:
            if 'pads' in l_attr:
                temp_pads = [0]*(len(l_attr['pads'])//2)
                for n in range(len(l_attr['pads'])//2):
                    temp_pads[n] = l_attr['pads'][n]+l_attr['pads'][n+len(temp_pads)]
                temp_kernel = [0]*4
                temp_kernel[1] = l_attr['kernel_shape'][0]-1
                temp_kernel[2] = l_attr['kernel_shape'][1]-1
                l_attr['output_shape'] = [int((x+z-k)/y) for k, x, y, z in zip(temp_kernel, l_attr['input_shape'], l_attr['strides'], temp_pads)]
            else:
                l_attr['output_shape'] = [int(x/y) for x, y in zip(l_attr['input_shape'], l_attr['strides'])]
        else:
            l_attr['output_shape'] = l_attr['input_shape']
        l_attr['output_shape'][-1] = l_attr['kernel_shape'][-1]
        logging.debug(l_attr)
        return deepcopy(l_attr)

    def compute_dims_base(self, l_name):
        l_attr = self.model_spec['layers'][l_name]
        l_type = l_attr['type'] 
        p_name = self.model_spec['layers'][l_name]['parents']
        #get input from parents
        if len(p_name) == 1:
            p_attr = self.model_spec['layers'][p_name[0]]
            l_attr['input_shape'] = p_attr['output_shape']
        elif len(p_name) > 1:
            if l_type in ['Add']:
                p_attr = self.model_spec['layers'][p_name[0]]
                l_attr['input_shape'] = p_attr['output_shape']
            elif l_type in ['Mul']:
                p_attr = self.model_spec['layers'][p_name[0]]
                l_attr['input_shape'] = p_attr['output_shape']
            elif l_type in ['Sub']:
                p_attr = self.model_spec['layers'][p_name[0]]
                l_attr['input_shape'] = p_attr['output_shape']
            else:
                print("help")
                print(l_type)
                raise NotImplementedError
        if 'strides' in l_attr:
            l_attr['output_shape'] = [int(x/y) for x, y in zip(l_attr['input_shape'], l_attr['strides'])]
        else:
            l_attr['output_shape'] = l_attr['input_shape']
        logging.debug(l_attr)
        return deepcopy(l_attr)


    def _check_left_parents(self, in_node_name, node):
        count = 0
        for in_edge in node['parents']:
            if in_node_name == in_edge.split(':')[0]:
                count += 1
        return count

    def to_json(self, filename):
        # replace in names, children and parents, :: with _
        od = OrderedDict([(":", ""), (" ", ""), ("/", ""), (".", "")])
        for node in self.model_spec['layers'].values():
            for i, in_edge in enumerate(node['parents']):
                node['parents'][i] = replace_all(in_edge, od)
            for i, out_edge in enumerate(node['children']):
                node['children'][i] = replace_all(out_edge, od)
        # replace in key names :: with _
        for key,value in self.model_spec['layers'].copy().items():
            self.model_spec['layers'][replace_all(key, od)] = self.model_spec['layers'].pop(key)


        with open(filename, 'w') as f:
            f.write(json.dumps(self.model_spec, indent=4))
            f.close()
        print("Stored to %s" % filename)
