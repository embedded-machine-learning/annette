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

    def __init__(self, name, json_file=None, in_dict=None, pad_pooling=True):
        self.model_spec = dict()
        self.model_spec['name'] = name
        self.model_spec['layers'] = dict()

        self.pad_pooling = pad_pooling

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
                    self.model_spec['input_layers'].append(c)
                self.model_spec['input_layers'].remove(name)

            if name in self.model_spec['output_layers']:
                for p in parents:
                    self.model_spec['output_layers'].append(p)
                self.model_spec['output_layers'].remove(name)

            del self.model_spec['layers'][name]
            self._get_topological_sort()
            return True
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
        # Get input from parents
        if 'axis' in l_attr:
            tmp_sum = 0
            l_attr['input_shape'] = [0]*len(p_name)
            for n, p_tmp in enumerate(p_name):
                p_attr = self.model_spec['layers'][p_tmp]
                tmp_sum = tmp_sum + p_attr['output_shape'][l_attr['axis']]
                l_attr['input_shape'][n] = copy(p_attr['output_shape'])
                l_attr['output_shape'] = copy(p_attr['output_shape'])
            l_attr['output_shape'][l_attr['axis']] = tmp_sum
            logging.debug(tmp_sum)
        else:
            raise RuntimeError('AnnetteGraph: Axis for concat layer is missing!')
            
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

    def compute_dims_MatMul(self, l_name):
        l_attr = self.model_spec['layers'][l_name]
        l_type = l_attr['type'] 
        p_name = self.model_spec['layers'][l_name]['parents']
        #get input from parents
        if len(p_name) == 1:
            p_attr = self.model_spec['layers'][p_name[0]]
            l_attr['input_shape'] = p_attr['output_shape']
        else:
            raise NotImplementedError

        l_attr['output_shape'][0] = l_attr['input_shape'][0] # set batch size eq. to parent
        logging.debug(l_attr)
        return deepcopy(l_attr)

    def compute_dims_Conv_and_Pool(self, l_name, is_2d=True, use_padding=True):
        l_attr = self.model_spec['layers'][l_name]
        l_type = l_attr['type']
        p_name = self.model_spec['layers'][l_name]['parents']
        #get input from parents
        if len(p_name) == 1:
            p_attr = self.model_spec['layers'][p_name[0]]
            l_attr['input_shape'] = p_attr['output_shape']
        else:
            raise NotImplementedError

        # Get or init relevant layer params:
        if l_type in ['Pool', 'Pool1d']: # kernel shape different for pooling operation
            kernel_w = l_attr['kernel_shape'][1]
            kernel_h = l_attr['kernel_shape'][2]
        else:
            kernel_w = l_attr['kernel_shape'][0]
            kernel_h = l_attr['kernel_shape'][1]
        strides = l_attr['strides'] if 'strides' in l_attr else [1]*len(l_attr['input_shape'])
        stride_w = strides[1]
        stride_h = strides[2]
        dilations = l_attr['dilations'] if 'dilations' in l_attr else [1]*len(l_attr['input_shape'])
        dilation_w = dilations[1]
        dilation_h = dilations[2]

        # calculate full padding (considering dilations) in both width and height dimensions:
        if not use_padding or 'pads' in l_attr.keys() and l_attr['pads'] == 'none':
            pad_w = pad_h = 0
        else:
            pad_w = (kernel_w + (kernel_w - 1) * (dilation_w - 1)) // 2
            pad_h = (kernel_h + (kernel_h - 1) * (dilation_h - 1)) // 2

        # set padding, dilation and stride in graph:
        l_attr['strides'] = strides
        l_attr['dilations'] = dilations
        if is_2d:
            l_attr['pads'] = [0, 0, pad_w, pad_w, pad_h, pad_h, 0, 0]
        else: # 1D Conv
            l_attr['pads'] = [0, 0, pad_w, pad_w, 0, 0]

        # in batch_size = out batch_size:
        l_attr['output_shape'] = [l_attr['input_shape'][0]]
        # out width:
        out_val = int((l_attr['input_shape'][1] + 2*pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w) + 1
        l_attr['output_shape'].append(out_val)
        # out height:
        if is_2d:
            out_val = int((l_attr['input_shape'][2] + 2*pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h) + 1
            l_attr['output_shape'].append(out_val)
        # out channels = kernel filters (conv) or input channels (pooling):
        if l_type in ['Pool', 'Pool1d']:
            l_attr['output_shape'].append(l_attr['input_shape'][-1])
        else:
            l_attr['output_shape'].append(l_attr['kernel_shape'][-1])

        logging.debug('compute_dims_{} output shape: {}'.format(l_type, l_attr['output_shape']))

        logging.debug(l_attr)
        return deepcopy(l_attr)

    def compute_dims_Glob_Pool(self, l_name, is_2d=True):
        l_attr = self.model_spec['layers'][l_name]
        p_name = self.model_spec['layers'][l_name]['parents']
        # Get input from parents:
        if len(p_name) == 1:
            p_attr = self.model_spec['layers'][p_name[0]]
            l_attr['input_shape'] = p_attr['output_shape']
        else:
            raise NotImplementedError

        # Set padding to 0 and stride to 1 in graph:
        l_attr['strides'] = [1]*len(l_attr['input_shape'])
        l_attr['pads'] = [0]*len(l_attr['input_shape'])*2

        # Set kernel size to spatial dimensions:
        l_attr['kernel_shape'][0] = 1
        l_attr['kernel_shape'][1] = l_attr['input_shape'][1]
        if is_2d:
            l_attr['kernel_shape'][2] = l_attr['input_shape'][2]
        l_attr['kernel_shape'][-1] = 1

        # in batch_size = out batch_size:
        l_attr['output_shape'] = [l_attr['input_shape'][0]]
        # global pooling results in spatial dimensions being 1:
        l_attr['output_shape'] += [1]*(len(l_attr['input_shape'])-2)
        # out channels = in channels:
        l_attr['output_shape'].append(l_attr['input_shape'][-1])

        logging.debug('compute_dims_global_pool output shape: {}'.format(l_attr['output_shape']))

        logging.debug(l_attr)
        return deepcopy(l_attr)

    def compute_dims_Conv(self, l_name):
        return self.compute_dims_Conv_and_Pool(l_name, is_2d=True)

    def compute_dims_Conv1d(self, l_name):
        return self.compute_dims_Conv_and_Pool(l_name, is_2d=False)

    def compute_dims_DepthwiseConv(self, l_name):
        return self.compute_dims_Conv_and_Pool(l_name, is_2d=True)

    def compute_dims_DepthwiseConv1d(self, l_name):
        return self.compute_dims_Conv_and_Pool(l_name, is_2d=False)

    def compute_dims_Pool(self, l_name):
        if self.model_spec['layers'][l_name]['kernel_shape'][1] == -1: # global pooling
            return self.compute_dims_Glob_Pool(l_name, is_2d=True)
        else: # local pooling
            return self.compute_dims_Conv_and_Pool(l_name, is_2d=True, use_padding=self.pad_pooling)

    def compute_dims_Pool1d(self, l_name):
        if self.model_spec['layers'][l_name]['kernel_shape'][1] == -1: # global pooling
            return self.compute_dims_Glob_Pool(l_name, is_2d=False)
        else: # local pooling
            return self.compute_dims_Conv_and_Pool(l_name, is_2d=False, use_padding=self.pad_pooling)

    def compute_dims_ConvTranspose2d(self, l_name, is_2d=True):
        l_attr = self.model_spec['layers'][l_name]
        l_type = l_attr['type']
        p_name = self.model_spec['layers'][l_name]['parents']
        #get input from parents
        if len(p_name) == 1:
            p_attr = self.model_spec['layers'][p_name[0]]
            l_attr['input_shape'] = p_attr['output_shape']
        else:
            raise NotImplementedError

        # Get or init relevant layer params:
        in_w = l_attr['input_shape'][1]
        in_h = l_attr['input_shape'][2]
        kernel_w = l_attr['kernel_shape'][0]
        kernel_h = l_attr['kernel_shape'][1]
        strides = l_attr['strides'] if 'strides' in l_attr else [1]*len(l_attr['input_shape'])
        stride_w = strides[1]
        stride_h = strides[2]
        dilations = l_attr['dilations'] if 'dilations' in l_attr else [1]*len(l_attr['input_shape'])
        dilation_w = dilations[1]
        dilation_h = dilations[2]

        # Calculate full padding:
        pad_w = kernel_w // 2
        pad_h = kernel_h // 2
        pad_w = (kernel_w + (kernel_w - 1) * (dilation_w - 1)) // 2
        pad_h = (kernel_h + (kernel_h - 1) * (dilation_h - 1)) // 2
        if is_2d:
            padding = [0, 0, pad_w, pad_w, pad_h, pad_h, 0, 0]
        else: # 1d
            padding = [0, 0, pad_w, pad_w, 0, 0]

        # Set output padding, s.t. input size = output size:
        if stride_w == 1:
            out_pad = [0, 0]
        elif stride_w == 2:
            out_pad = [1, 1]
        else:
            out_pad = [stride_w - 2, stride_w - 2]

        # set padding, dilation and stride in graph:
        l_attr['strides'] = strides
        l_attr['dilations'] = dilations
        l_attr['pads'] = padding
        l_attr['out_pad'] = out_pad

        # in batch_size = out batch_size:
        l_attr['output_shape'] = [l_attr['input_shape'][0]]
        # out width:
        out_val = (in_w - 1)*stride_w - 2*pad_w + dilation_w*(kernel_w - 1) + out_pad[0] + 1
        l_attr['output_shape'].append(out_val)
        # out height:
        if is_2d:
            out_val = (in_h - 1)*stride_h - 2*pad_h + dilation_h*(kernel_h - 1) + out_pad[1] + 1
            l_attr['output_shape'].append(out_val)
        # out channels = kernel filters:
        l_attr['output_shape'].append(l_attr['kernel_shape'][-1])

        logging.debug('compute_dims_{} output shape: {}'.format(l_type, l_attr['output_shape']))

        logging.debug(l_attr)
        return deepcopy(l_attr)

    def compute_dims_ConvTranspose1d(self, l_name):
        return self.compute_dims_ConvTranspose2d(l_name, is_2d=False)

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
            else:
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
        with open(filename, 'w') as f:
            f.write(json.dumps(self.model_spec, indent=4))
        print("Stored to %s" % filename)
