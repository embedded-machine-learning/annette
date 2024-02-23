"""Annette ONNX Graph Utils

This Module contains the Graph Utilities to generate Annette readable graphs from ONNX
or read directly from json.

Code based on https://github.com/waltYeh/onnx2keras
"""
from __future__ import print_function

import json
import logging
import sys
from onnx import numpy_helper
import onnx
import numpy as np
from functools import reduce
from .annette_graph import AnnetteGraph
import copy

__author__ = "Matthias Wess"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache-2.0"

logger = logging.getLogger(__name__)

ONNX2ANNETTE = {
    "Clip": "Relu6",
    "BatchNormalization": "BatchNorm"
    }

def nhwc2nchw(in_shape):
    out_shape = [in_shape[i] for i in [0,3,1,2]]
    return out_shape

def nchw2nhwc(in_shape):
    logger.debug(in_shape)
    out_shape = [in_shape[i] for i in [0,2,3,1]]
    logger.debug(out_shape)
    return out_shape

def onnx_node_by_name(name, onnx_nodes):
    for node_index, node in enumerate(onnx_nodes):
        node_name = node.name
        if node_name == name:
            print(node_name)

def onnx_node_attributes_to_dict(args):
    """
    Parse ONNX attributes to Python dictionary
    :param args: ONNX attributes object
    :return: Python dictionary
    """
    def onnx_attribute_to_dict(onnx_attr):
        """
        Parse ONNX attribute
        :param onnx_attr: ONNX attribute
        :return: Python data type
        """
        if onnx_attr.HasField('t'):
            return numpy_helper.to_array(getattr(onnx_attr, 't'))

        for attr_type in ['f', 'i', 's']:
            if onnx_attr.HasField(attr_type):
                return getattr(onnx_attr, attr_type)

        for attr_type in ['floats', 'ints', 'strings']:
            if getattr(onnx_attr, attr_type):
                return list(getattr(onnx_attr, attr_type))
    return {arg.name: onnx_attribute_to_dict(arg) for arg in args}


class ONNXGraph:

    """ ONNX Graph Class

    Args
        graphfile (str): MMDNN graphfile
        weightfile(str, optional): MMDNN weightfile, dropped anyways

    Attributes:
        IR_graph : mmdnn Intermediate Representation Graph 
    """

    def __init__(self, graphfile, input_names=None):
        print("Initializing network...")

        self.graphfile = graphfile
        self.input_names = input_names

        self.available = {
            'Conv': self.convert_conv,
            'Mul': self.convert_base,
            'Add': self.convert_base,
            'Sub': self.convert_base,
            'Pow': self.convert_base,
            'LeakyRelu': self.convert_base,
            'Clip': self.convert_base,
            'Shape': self.convert_shape,
            'Gather': self.convert_gather,
            'Unsqueeze': self.convert_base,
            'Concat': self.convert_concat,
            'Constant': self.convert_none,
            'Reshape': self.convert_reshape,
            'Resize': self.convert_resize,
            'Flatten': self.convert_flatten,
            'Dropout': self.convert_base,
            'Softmax': self.convert_base,
            'Relu': self.convert_base,
            'Gemm': self.convert_gemm,
            'LRN': self.convert_LRN,
            'BatchNormalization': self.convert_base,
            'MaxPool': self.convert_pool,
            'GlobalAveragePool': self.convert_pool,
            'Less': self.convert_base,
            'Div' : self.convert_base,
            'Sigmoid': self.convert_base,
            'Slice': self.convert_slice,
            'ReduceMean': self.convert_reducemean,
            'Transpose': self.convert_transpose}

        self.onnx_model = onnx.load(graphfile)

        print("ONNX Network loaded successfully...\n")

    def onnx_to_annette(self, network, input_names, input_shapes=None, name_policy=None, verbose=True, change_ordering=False):
        logger.info('Converter is called.')
        self.input_names = input_names
        self.annette_graph = AnnetteGraph(network)

        onnx_weights = self.onnx_model.graph.initializer
        onnx_inputs = self.onnx_model.graph.input
        onnx_outputs = [i.name for i in self.onnx_model.graph.output]
        onnx_nodes = self.onnx_model.graph.node

        logger.debug('List input shapes:')
        logger.debug(input_shapes)

        logger.debug('List inputs:')
        for i, input in enumerate(onnx_inputs):
            logger.debug('Input {0} -> {1}'.format(i, input.name))

        logger.debug('List outputs:')
        for i, output in enumerate(onnx_outputs):
            logger.debug('Output {0} -> {1}'.format(i, output))

        logger.debug('Gathering weights to dictionary.')
        weights = {}
        for onnx_w in onnx_weights:
            try:
                if len(onnx_w.ListFields()) < 4:
                    onnx_extracted_weights_name = onnx_w.ListFields()[1][1]
                else:
                    onnx_extracted_weights_name = onnx_w.ListFields()[2][1]
                weights[onnx_extracted_weights_name] = numpy_helper.to_array(onnx_w)
            except:
                onnx_extracted_weights_name = onnx_w.ListFields()[3][1]
                weights[onnx_extracted_weights_name] = numpy_helper.to_array(onnx_w)

            logger.debug('Found weight {0} with shape {1}.'.format(
                        onnx_extracted_weights_name,
                        weights[onnx_extracted_weights_name].shape))
    #    print(weights)
        layers = dict()
        annette_outputs = []
        annette_inputs = []

        print(type(self.input_names),self.input_names)
        if (type(self.input_names) == str):
            self.input_names = [self.input_names]

        if self.input_names:
            for i, input_name in enumerate(self.input_names):
                for onnx_i in onnx_inputs:
                    print(onnx_i.name)
                    print(input_name)
                    if onnx_i.name == input_name:
                        if input_shapes:
                            input_shape = input_shapes[i]
                        else:
                            input_shape = [i.dim_value for i in onnx_i.type.tensor_type.shape.dim][1:]

                        print("#TODO add input layer",input_shape,input_name)
                        """
                        layers[input_name] = keras.layers.InputLayer(
                            input_shape=input_shape, name=input_name
                        ).output
                        """
                        layers[input_name] = {'type':'DataInput',
                            'parents':[],
                            'children':[],
                            'output_shape':[-1] + input_shape}
                        self.annette_graph.add_layer(input_name, layers[input_name],True)

                        annette_inputs.append(layers[input_name])

                        logger.debug('Found input {0} with shape {1}'.format(input_name, input_shape))
        if len(layers) == 0:
            print("Inputs '%s' not found! Probable inputs:" % self.input_names)
            for i, input in enumerate(onnx_inputs):
                print('Input {0} -> {1}'.format(i, input.name))
                return 0

        self.weights = weights
        # Convert every operation separable
        node_names = []
        for node_index, node in enumerate(onnx_nodes):
            node_type = node.op_type
            node_params = onnx_node_attributes_to_dict(node.attribute)

            # Add global converter info:
            node_params['change_ordering'] = change_ordering
            node_params['name_policy'] = name_policy

            node_name = str(node.output[0])
            print (node_type)
            print(node_name)
            annette_names = []
            for output_index, output in enumerate(node.output):
                if name_policy == 'short':
                    annette_name = keras_name_i = str(output)[:8]
                    suffix = 1
                    while annette_name_i in node_names:
                        annette_name_i = keras_name + '_' + str(suffix)
                        suffix += 1
                    annette_names.append(annette_name_i)
                elif name_policy == 'renumerate':
                    postfix = node_index if len(node.output) == 1 else "%s_%s" % (node_index, output_index)
                    annette_names.append('LAYER_%s' % postfix)
                else:
                    annette_names.append(output)

            if len(node.output) != 1:
                logger.warning('Trying to convert multi-output node')
                node_params['_outputs'] = list(node.output)
                node_names.extend(annette_names)
            else:
                annette_names = annette_names[0]
                node_names.append(annette_names)

            logger.debug('######')
            logger.debug('...')
            logger.debug('Converting ONNX operation')
            logger.debug('type: %s', node_type)
            logger.debug('node_name: %s', node_name)
            logger.debug('node_params: %s', node_params)
            logger.debug('...')

            logger.debug('Check if all inputs are available:')
            if len(node.input) == 0 and node_type != 'Constant':
                raise AttributeError('Operation doesn\'t have an input. Aborting.')

            for i, node_input in enumerate(node.input):
                logger.debug('Check input %i (name %s).', i, node_input)
                if node_input not in layers:
                    logger.debug('The input not found in layers / model inputs.')

                    if node_input in weights:
                        logger.debug('Found in weights, add as a numpy constant.')
                        #layers[node_input] = weights[node_input]
                    else:
                        print(node)
                        #raise AttributeError('Current node is not in weights / model inputs / layers.')
            #else:
            #   logger.debug('... found all, continue')

            self.available[node_type](node, node_params, layers, node_name, annette_names)

            logging.debug(self.annette_graph.model_spec)
            print('used converter once')
        print('finished layers')


        for l_n, l_attr in self.annette_graph.model_spec['layers'].items():
            print(l_n)
            print(l_attr)
            for s in ['type']:
                if l_attr[s] in ONNX2ANNETTE.keys():
                    self.annette_graph.model_spec['layers'][l_n][s] = ONNX2ANNETTE[l_attr[s]]
                    
            for s in ['input_shape','output_shape']:
                try:
                    if len(l_attr[s]) == 4:
                        res =  nchw2nhwc(l_attr[s])
                        print(res)
                        self.annette_graph.model_spec['layers'][l_n][s] = nchw2nhwc(l_attr[s])
                except:
                    logger.debug('No shape')


        logging.debug(self.annette_graph.__dict__)
        return self.annette_graph

    def convert_conv(self, node, params, layers, node_name, annette_name):
        """
        Convert convolution layer
        :param node: current operation node
        :param params: operation attributes
        :param layers: available annette layers
        :param node_name: internal converter name
        :param keras_name: resulting layer name
        :return: None
        """
        logger = logging.getLogger('onnx2annette:conv')
        input_0 = node.input[0]
        if len(node.input) == 3:
            logger.debug('Conv with bias')
            # Has bias
            has_bias = True
            W = self.weights[node.input[1]]
            bias = self.weights[node.input[2]]

        elif len(node.input) == 2:
            logger.debug('Conv without bias')
            has_bias = False
            W = self.weights[node.input[1]]
            bias = None

        else:
            raise NotImplementedError('Not implemented')
        n_groups = params['group'] if 'group' in params else 1
        print('n_groups')
        print(n_groups)
        dilation = params['dilations'][0] if 'dilations' in params else 1
        pads = params['pads'] if 'pads' in params else [0, 0, 0]
        strides = params['strides'] if 'strides' in params else [1, 1, 1]

        if len(W.shape) == 4:  # 2D conv
            logger.debug('2D convolution')
            
            padding = None
            if len(pads) == 2 and (pads[0] > 0 or pads[1] > 0):
                padding = [pads[0], pads[1], pads[0], pads[1]]
            elif len(pads) == 4 and (pads[0] > 0 or pads[1] > 0 or pads[2] > 0 or pads[3] > 0):
                padding = [pads[0], pads[1], pads[2], pads[3]]

            #if padding:
            #    raise NotImplementedError('Not implemented')

            W = W.transpose(2, 3, 1, 0)
            height, width, channels_per_group, out_channels = W.shape
            #3,3,3,32
            in_channels = channels_per_group * n_groups

            print(height, width, channels_per_group, out_channels)

            logger.debug('Out Channels %s' % out_channels)#32
            logger.debug((height, width))#(3,3)
            logger.debug((strides[0], strides[1]))#(1,1)
            logger.debug('Bias %s' % has_bias)#false
            print(dilation)#1
            print(annette_name)#convolut
            kernel_shape = [height, width, channels_per_group, out_channels]
            strides = [1, strides[0], strides[1], 1]
            pad = [0]*8
            if 'auto_pad' in params.keys() and params['auto_pad'] == 'SAME_UPPER':
                print('autopad')
                if strides == [1, 1, 1, 1]:
                    p_h = int((height-1)/2)
                    p_w = int((width-1)/2)
                    pad[1] = pad[5] = p_h
                    pad[2] = pad[6] = p_w
                    print(p_h)
                    print(p_w)
                else:
                    pad[1] = pad[5] = padding[0]
                    pad[2] = pad[6] = padding[1]
                    #raise NotImplementedError('Not implemented')
            in_shape = layers[input_0]['output_shape']
            if (len(in_shape) == 4):
                out_shape = [in_shape[0], out_channels, int(in_shape[2]/strides[1]), int(in_shape[3]/strides[2])]
            elif (len(in_shape) == 2):
                out_shape = [in_shape[0], out_channels]
            else:
                raise NotImplementedError

            logger.debug("input_shape: %s" % in_shape)
            logger.debug("output_shape: %s" % out_shape)
            logger.debug("kernel_shape: %s" % kernel_shape)
            logger.debug("strides: %s" % strides)
            logger.debug("pads: %s" % pad)
            layers[node_name] = {'type':'Conv',
                'parents': [input_0],
                'children':[],
                'input_shape': in_shape,
                'strides': strides,
                'pads': pad,
                'kernel_shape': kernel_shape,
                'output_shape': out_shape}
            self.annette_graph.add_layer(node_name, layers[node_name],True)
            
            logger.debug("layers: %s" % layers)
            #using functional model
            #input 0: Tensor("input_1:0", shape=(None, 3, 0, 0), dtype=float32)
        else:
            raise NotImplementedError('Not implemented')



        return

    def convert_transpose(self, node, params, layers, node_name, annette_name):
        """
        Convert convolution layer
        :param node: current operation node
        :param params: operation attributes
        :param layers: available annette layers
        :param node_name: internal converter name
        :param keras_name: resulting layer name
        :return: None
        """
        logger = logging.getLogger('onnx2annette:transp')
        print("node:",node)
        #print("params:",params)
        print(layers)
        print(node_name)
        print(annette_name)
        input_name = node.input[0]

        
        if params['perm'][0] != 0:
            logger.warning('Can\'t permute batch dimension. Result may be wrong.')
            raise NotImplementedError('Can\'t modify this type of data')
        else:
            print(layers[input_name])
            in_shape = layers[input_name]['output_shape']
            out_shape = [in_shape[i] for i in params['perm']]
            layers[node_name] = {'type':'Transpose',
                'parents': [input_name],
                'children':[],
                'input_shape': layers[input_name]['output_shape'],
                'perm': params['perm'],
                'output_shape': out_shape}
            self.annette_graph.add_layer(node_name, layers[node_name],True)

        return

    def convert_LRN(self, node, params, layers, node_name, annette_name):
        """
        Convert LRN Layer 
        :param node: current operation node
        :param params: operation attributes
        :param layers: available annette layers
        :param node_name: internal converter name
        :param keras_name: resulting layer name
        :return: None
        """
        logger = logging.getLogger('onnx2annette:LRN')
        print("node:",node)
        #print("params:",params)
        #print(layers)
        print(node_name)
        print(annette_name)
        input_name = node.input[0]

        
        layers[node_name] = {'type':'LRN',
            'parents': [input_name],
            'children':[],
            'input_shape': layers[input_name]['output_shape'],
            'output_shape': layers[input_name]['output_shape']
            }
        self.annette_graph.add_layer(node_name, layers[node_name],True)

        return

    def convert_mul(self, node, params, layers, node_name, annette_name):
        """
        Convert multiplication layer
        :param node: current operation node
        :param params: operation attributes
        :param layers: available annette layers
        :param node_name: internal converter name
        :param keras_name: resulting layer name
        :return: None
        """
        logger = logging.getLogger('onnx2annette:mul')
        print("node:",node)
        #print("params:",params)
        print(layers)
        print(node_name)
        print(annette_name)
        input_name = node.input[0]

        
        layers[node_name] = {'type':'Mul',
            'parents': [input_name],
            'children':[],
            'input_shape': layers[input_name]['output_shape'],
            'output_shape': layers[input_name]['output_shape']
            }
        self.annette_graph.add_layer(node_name, layers[node_name],True)

        return

    def convert_reducemean(self, node, params, layers, node_name, annette_name):
        """
        Convert reducemena layer
        :param node: current operation node
        :param params: operation attributes
        :param layers: available annette layers
        :param node_name: internal converter name
        :param keras_name: resulting layer name
        :return: None
        """
        logger = logging.getLogger('onnx2annette:reducemean')
        print("node:",node)
        #print("params:",params)
        print(layers)
        print(node_name)
        print(annette_name)
        input_name = node.input[0]

        in_shape = layers[input_name]['output_shape']
        out_shape = copy.copy(in_shape)
        for p in params['axes']:
            out_shape[p] = 1
        print(params['axes'])

        logger.debug("input_shape: %s" % in_shape)
        logger.debug("output_shape: %s" % out_shape)

        layers[node_name] = {'type':'ReduceMean',
            'parents': [input_name],
            'children':[],
            'input_shape': in_shape,
            'output_shape': out_shape,
            }

        self.annette_graph.add_layer(node_name, layers[node_name],True)

        return

    def convert_pool(self, node, params, layers, node_name, annette_name):
        """
        Convert pool layer
        :param node: current operation node
        :param params: operation attributes
        :param layers: available annette layers
        :param node_name: internal converter name
        :param keras_name: resulting layer name
        :return: None
        """
        logger = logging.getLogger('onnx2annette:mul')
        print("node:",node)
        #print("params:",params)
        print(layers)
        print(node_name)
        print(annette_name)
        input_name = node.input[0]

        pads = params['pads'] if 'pads' in params else [0, 0]
        strides = params['strides'] if 'strides' in params else [1, 1]
        kernel_shape = params['kernel_shape'] if 'kernel_shape' in params else [1, 1]

        kernel_shape = [1 ,kernel_shape[0], kernel_shape[1], 1]
        strides = [1, strides[0], strides[1], 1]
        pads = [0]*8
        if 'auto_pad' in params.keys():
            if strides == [1, 1, 1, 1] or strides == kernel_shape:
                p_h = int((kernel_shape[1]-1)/2)
                p_w = int((kernel_shape[0]-1)/2)
                pads[1] = pads[5] = p_h
                pads[2] = pads[6] = p_w
            else:
                raise NotImplementedError('Not implemented')
        in_shape = layers[input_name]['output_shape']
        out_shape = [in_shape[0], in_shape[1], int(in_shape[2]/strides[1]), int(in_shape[3]/strides[2])]
        if node.op_type == 'MaxPool':
            pooling_type = 'MAX' 
        elif node.op_type == 'AvgPool':
            pooling_type = 'AVG' 
        elif node.op_type == 'GlobalAveragePool':
            pooling_type = 'AVG' 
            pads = [0]*8
            strides = [1]*4
            in_shape = layers[input_name]['output_shape']
            kernel_shape = [1, in_shape[2], in_shape[3], 1]
            out_shape = [in_shape[0], in_shape[1], 1, 1]
        else:
            raise NotImplementedError('Not implemented')

        logger.debug("input_shape: %s" % in_shape)
        logger.debug("output_shape: %s" % out_shape)
        logger.debug("kernel_shape: %s" % kernel_shape)
        logger.debug("strides: %s" % strides)
        logger.debug("pads: %s" % pads)
        logger.debug("pooling_type: %s" % pooling_type)

        layers[node_name] = {'type':'Pool',
            'parents': [input_name],
            'children':[],
            'input_shape': in_shape,
            'strides': strides,
            'pads': pads,
            'kernel_shape': kernel_shape,
            'output_shape': out_shape,
            'pooling_type': pooling_type
            }

        self.annette_graph.add_layer(node_name, layers[node_name],True)

        return

    def convert_concat(self, node, params, layers, node_name, annette_name):
        """
        Convert concat layer
        :param node: current operation node
        :param params: operation attributes
        :param layers: available annette layers
        :param node_name: internal converter name
        :param keras_name: resulting layer name
        :return: None
        """
        logger = logging.getLogger('onnx2annette:reshape')
        print("node:",node)
        print(layers)
        print(node_name)
        print(annette_name)
        print(params)

        ins = []
        for i in node.input:
            print(i)
            if i in layers.keys():
                ins.append(i)
        if len(ins) != 1:
            assert AttributeError('More than 1 input layer')
            #TODO check if all shapes similar then presume elemwise
            input_name = ins[0]
        else:
            input_name = ins[0]
        
        output_shape = copy.deepcopy(layers[ins[0]]['output_shape'])
        output_shape[params['axis']] = 0
        for i in ins:
            output_shape[params['axis']] += layers[i]['output_shape'][params['axis']]

        layers[node_name] = {'type':node.op_type,
            'parents': ins,
            'children':[],
            'input_shape': layers[input_name]['output_shape'],
            'output_shape': output_shape
            }
        self.annette_graph.add_layer(node_name, layers[node_name],True)

        return
        
    def convert_reshape(self, node, params, layers, node_name, annette_name):
        """
        Convert reshape layer
        :param node: current operation node
        :param params: operation attributes
        :param layers: available annette layers
        :param node_name: internal converter name
        :param keras_name: resulting layer name
        :return: None
        """
        logger = logging.getLogger('onnx2annette:reshape')
        print("node:",node)
        print(layers)
        print(node_name)
        print(annette_name)
        print(self.weights.keys())

        ins = []
        for i in node.input:
            print(i)
            if i in layers.keys():
                ins.append(i)
        if len(ins) != 1:
            assert AttributeError('More than 1 input layer')
            #TODO check if all shapes similar then presume elemwise
            input_name = ins[0]
        else:
            input_name = ins[0]
        
        input_0 = layers[node.input[0]]
        output_shape = self.weights[node.input[1]].tolist()
        input_shape = layers[input_name]['output_shape']
        print(input_shape)
        prod = reduce(lambda x, y: x * y, layers[input_name]['output_shape'])
        prod_out = reduce(lambda x, y: x * y, output_shape)
        for n, x in enumerate(output_shape):
            if n == 0: 
                if input_shape[0] == -1 and output_shape[0] == 1:
                    prod = prod*-1
                    output_shape[0] = -1
            elif x == -1:
                output_shape[n] = int(prod/prod_out*-1)
                print(output_shape)


        
        print(input_0)
        print(output_shape)

        layers[node_name] = {'type':node.op_type,
            'parents': ins,
            'children':[],
            'input_shape': layers[input_name]['output_shape'],
            'output_shape': output_shape
            }
        self.annette_graph.add_layer(node_name, layers[node_name],True)

        return

    def convert_base(self, node, params, layers, node_name, annette_name):
        """
        Convert base layer
        :param node: current operation node
        :param params: operation attributes
        :param layers: available annette layers
        :param node_name: internal converter name
        :param keras_name: resulting layer name
        :return: None
        """
        logger = logging.getLogger('onnx2annette:base')
        print("node:",node)
        print(layers)
        print(node_name)
        print(annette_name)

        ins = []
        for i in node.input:
            print(i)
            if i in layers.keys():
                ins.append(i)
        if len(ins) != 1:
            assert AttributeError('More than 1 input layer')
            #TODO check if all shapes similar then presume elemwise
            input_name = ins[0]
        else:
            input_name = ins[0]

        layers[node_name] = {'type':node.op_type,
            'parents': ins,
            'children':[],
            'input_shape': layers[input_name]['output_shape'],
            'output_shape': layers[input_name]['output_shape']
            }
        self.annette_graph.add_layer(node_name, layers[node_name],True)

        return

    def convert_slice(self, node, params, layers, node_name, annette_name):
        """
        Convert slice layer
        :param node: current operation node
        :param params: operation attributes
        :param layers: available annette layers
        :param node_name: internal converter name
        :param keras_name: resulting layer name
        :return: None
        """
        logger = logging.getLogger('onnx2annette:slice')
        print("node:",node)
        print(node_name)
        print(annette_name)
        print(params)

        ins = []
        for i in node.input:
            print(i)
            if i in layers.keys():
                ins.append(i)
        print(ins)
        if len(ins) != 1:
            assert AttributeError('More than 1 input layer')
            #TODO check if all shapes similar then presume elemwise
            input_name = ins[0]
        else:
            input_name = ins[0]


        starts = self.weights[node.input[1]].tolist()
        ends = self.weights[node.input[2]].tolist()
        axes = self.weights[node.input[3]].tolist()
        steps = self.weights[node.input[4]].tolist()
        input_shape = copy.deepcopy(layers[input_name]['output_shape'])
        print(starts,ends,axes,steps)
        output_shape = input_shape
        if ends[0] != 9223372036854775807:
            diff = ends[0] - starts[0]
        else:
            diff = input_shape[axes[0]] - starts[0]

        output_shape[axes[0]] = int(np.ceil((diff)/steps[0]))


        layers[node_name] = {'type':node.op_type,
            'parents': ins,
            'children':[],
            'input_shape': layers[input_name]['output_shape'],
            'output_shape': output_shape
            }
        self.annette_graph.add_layer(node_name, layers[node_name],True)

        return

    def convert_gemm(self, node, params, layers, node_name, annette_name):
        """
        Convert gemm layer
        :param node: current operation node
        :param params: operation attributes
        :param layers: available annette layers
        :param node_name: internal converter name
        :param keras_name: resulting layer name
        :return: None
        """
        logger = logging.getLogger('onnx2annette:base')
        print("node:",node)
        print(layers)
        print(node_name)
        print(annette_name)

        ins = []
        for i in node.input:
            print(i)
            if i in layers.keys():
                ins.append(i)
        if len(ins) != 1:
            assert AttributeError('More than 1 input layer')
            #TODO check if all shapes similar then presume elemwise
            input_name = ins[0]
        else:
            input_name = ins[0]
        print(ins)

        print(layers.keys())
        # Check if Bias available
        if len(node.input) == 3:
            has_bias = True
            annette_weights = [self.weights[node.input[1]], self.weights[node.input[2]]]
            logger.debug('Convert GEMM with bias.')
        elif len(node.input) == 2:
            has_bias = False
            annette_weights= [self.weights[node.input[1]]]
            logger.debug('Convert GEMM without bias.')
        else:
            raise AttributeError('More than 3 or less than 2 inputs')

        # Linear can have additional flag to transpose weights
        if 'transB' in params and params['transB'] == 1:
            logger.debug('Transposing W matrix.')
            annette_weights[0] = annette_weights[0].transpose()

        # Estimate input/output neurons
        input_channels, output_channels = annette_weights[0].shape
        logger.debug('Input units %s, output units %s.', input_channels, output_channels)

        """
        dense = keras.layers.Dense(
            output_channels,
            weights=keras_weights, name=keras_name, bias_initializer='zeros', kernel_initializer='zeros', use_bias=has_bias
        )
        """

        # The first input - always X
        """
        try:
            layers[node_name] = dense(layers[node.input[0]])
        except ValueError:
            reshape = keras.layers.Reshape([input_channels], name=keras_name + '_reshape')
            reshaped_x = reshape(layers[node.input[0]])
            layers[node_name] = dense(reshaped_x)
        """

        layers[node_name] = {'type': 'FullyConnected',
            'parents': ins,
            'children':[],
            'input_shape': [layers[input_name]['output_shape'][0], input_channels],
            'output_shape': [layers[input_name]['output_shape'][0], output_channels]
            }
        self.annette_graph.add_layer(node_name, layers[node_name],True)

        return

    def convert_flatten(self, node, params, layers, node_name, annette_name):
        """
        Convert shape layer
        :param node: current operation node
        :param params: operation attributes
        :param layers: available annette layers
        :param node_name: internal converter name
        :param keras_name: resulting layer name
        :return: None
        """
        logger = logging.getLogger('onnx2annette:shape')
        print("node:",node)
        print(layers)
        print(node_name)
        print(annette_name)

        ins = []
        for i in node.input:
            print(i)
            if i in layers.keys():
                ins.append(i)
        if len(ins) != 1:
            assert AttributeError('More than 1 input layer')
            #TODO check if all shapes similar then presume elemwise
            input_name = ins[0]
        else:
            input_name = ins[0]
        print(ins)

        output_shape = [layers[input_name]['output_shape'][0], reduce(lambda x, y: x*y, layers[input_name]['output_shape'][1:])]


        layers[node_name] = {'type':node.op_type,
            'parents': ins,
            'children':[],
            'input_shape': layers[input_name]['output_shape'],
            'output_shape': output_shape,
            }
        self.annette_graph.add_layer(node_name, layers[node_name],True)

        return

    def convert_resize(self, node, params, layers, node_name, annette_name):
        """
        Convert resize layer
        :param node: current operation node
        :param params: operation attributes
        :param layers: available annette layers
        :param node_name: internal converter name
        :param keras_name: resulting layer name
        :return: None
        """
        logger = logging.getLogger('onnx2annette:resize')
        logger.debug("node:",node)
        logger.debug(layers)
        logger.debug(node_name)
        logger.debug(annette_name)

        ins = []
        for i in node.input:
            print(i)
            if i in layers.keys():
                ins.append(i)
        if len(ins) != 1:
            assert AttributeError('More than 1 input layer')
            #TODO check if all shapes similar then presume elemwise
            input_name = ins[0]
        else:
            input_name = ins[0]

        #onnx_node_by_name('562', self.onnx_model.graph.node)
        logger.debug(node.input)
        #onnx_node_attributes_to_dict(self.)
        #logger.debug(self.onnx_model.graph.node[10])
        output_shape = [int(x*y) for x, y in zip(layers[input_name]['output_shape'],self.weights[node.input[2]])]
        logger.debug(output_shape)

        layers[node_name] = {'type':node.op_type,
            'parents': ins,
            'children':[],
            'input_shape': layers[input_name]['output_shape'],
            'output_shape': output_shape,
            }
        self.annette_graph.add_layer(node_name, layers[node_name],True)

        return

    def convert_gather(self, node, params, layers, node_name, annette_name):
        """
        Convert shape layer
        :param node: current operation node
        :param params: operation attributes
        :param layers: available annette layers
        :param node_name: internal converter name
        :param keras_name: resulting layer name
        :return: None
        """
        logger = logging.getLogger('onnx2annette:shape')
        print("node:",node)
        print(layers)
        print(node_name)
        print(annette_name)

        ins = []
        for i in node.input:
            print(i)
            if i in layers.keys():
                ins.append(i)
        if len(ins) != 1:
            assert AttributeError('More than 1 input layer')
            #TODO check if all shapes similar then presume elemwise
            input_name = ins[0]
        else:
            input_name = ins[0]
        print(ins)


        layers[node_name] = {'type':node.op_type,
            'parents': ins,
            'children':[],
            'input_shape': layers[input_name]['output_shape'],
            'output_shape': [len(layers[input_name]['output_shape'])]
            }
        self.annette_graph.add_layer(node_name, layers[node_name],True)

        return

    def convert_shape(self, node, params, layers, node_name, annette_name):
        """
        Convert shape layer
        :param node: current operation node
        :param params: operation attributes
        :param layers: available annette layers
        :param node_name: internal converter name
        :param keras_name: resulting layer name
        :return: None
        """
        logger = logging.getLogger('onnx2annette:shape')
        print("node:",node)
        print(layers)
        print(node_name)
        print(annette_name)

        ins = []
        for i in node.input:
            print(i)
            if i in layers.keys():
                ins.append(i)
        if len(ins) != 1:
            assert AttributeError('More than 1 input layer')
            #TODO check if all shapes similar then presume elemwise
            input_name = ins[0]
        else:
            input_name = ins[0]
        print(ins)


        layers[node_name] = {'type':node.op_type,
            'parents': ins,
            'children':[],
            'input_shape': layers[input_name]['output_shape'],
            'output_shape': [len(layers[input_name]['output_shape'])]
            }
        self.annette_graph.add_layer(node_name, layers[node_name],True)

        return

    def convert_none(self, node, params, layers, node_name, annette_name):
        """
        Convert none layer
        :param node: current operation node
        :param params: operation attributes
        :param layers: available annette layers
        :param node_name: internal converter name
        :param keras_name: resulting layer name
        :return: None
        """
        logger = logging.getLogger('onnx2annette:base')

        return