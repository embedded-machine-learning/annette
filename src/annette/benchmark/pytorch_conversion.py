from collections import OrderedDict
import logging
import torch
import torch.nn as nn


class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.add(a, b)
    

class Mul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.mul(a, b)


class Resize(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, input):
        return torch.nn.functional.interpolate(input, size=self.size, mode='nearest')


class Flatten(nn.Module):
    def __init__(self, start_dim=0, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        # return torch.flatten(input, self.start_dim, self.end_dim)

        # Use reshape instead of flatten to fix ONNX export problems:
        return torch.reshape(input, (1, -1))

class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, *inputs):
        return torch.cat(inputs, dim=self.dim)


class Squeeze(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return torch.squeeze(input)
    

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return torch.reshape(input, self.shape)


class TorchGraph(nn.Module):
    '''
    PyTorch Module created on top of an Annette graph instance.
    Conversion from Annette format to PyTorch is done automatically on initialization.

    Parameters
    ----------
    annette_graph : AnnetteGraph
        Annette graph instance of model to be built
    '''
    def __init__(self, annette_graph):
        super().__init__()
        self.output_layers = annette_graph.model_spec['output_layers']
        self.nodes = nn.ModuleDict()
        self.parent_nodes = OrderedDict()
        for layer_name in annette_graph.topological_sort:
            layer_info = annette_graph.model_spec['layers'][layer_name]
            layer_type = layer_info['type']
            if layer_type in ['Conv', 'DepthwiseConv']:
                kernel_size = tuple(layer_info['kernel_shape'][0:2][::-1])
                in_channels = layer_info['input_shape'][3]
                out_channels = layer_info['output_shape'][3]
                strides = tuple(layer_info['strides'][1:3][::-1])
                #padding = tuple([layer_info['pads'][4], layer_info['pads'][2]])
                padding = tuple([layer_info['kernel_shape'][0]//2, layer_info['kernel_shape'][1]//2])
                # if no dilations are specified, the default is 1
                if 'dilations' not in layer_info:
                    dilations = (1, 1)
                else:
                    dilations = tuple(layer_info['dilations'][1:3][::-1])
                # groups = in_channels => depthwise conv
                groups = in_channels if layer_type == 'DepthwiseConv' else 1
                assert groups == 1 or in_channels == out_channels
                self.nodes[layer_name] = nn.Conv2d(in_channels,
                                                   out_channels,
                                                   kernel_size,
                                                   strides,
                                                   padding,
                                                   dilation=dilations,
                                                   groups=groups)
            elif layer_type in ['Conv1d', 'DepthwiseConv1d']:
                kernel_size = layer_info['kernel_shape'][0]
                in_channels = layer_info['input_shape'][2]
                out_channels = layer_info['output_shape'][2]
                stride = layer_info['strides'][1]
                padding = layer_info['pads'][2]
                dilation = layer_info['dilations'][1]
                # groups = in_channels => depthwise conv
                groups = in_channels if layer_type == 'DepthwiseConv' else 1
                assert groups == 1 or in_channels == out_channels
                self.nodes[layer_name] = nn.Conv1d(in_channels,
                                                   out_channels,
                                                   kernel_size,
                                                   stride,
                                                   padding,
                                                   dilation=dilation,
                                                   groups=groups)
            elif layer_type == 'ConvTranspose2d':
                kernel_size = tuple(layer_info['kernel_shape'][0:2][::-1])
                in_channels = layer_info['input_shape'][3]
                out_channels = layer_info['output_shape'][3]
                strides = tuple(layer_info['strides'][1:3][::-1])
                padding = tuple([layer_info['pads'][4], layer_info['pads'][2]])
                dilations = tuple(layer_info['dilations'][1:3][::-1])
                out_pad = layer_info['out_pad'][::-1] if 'out_pad' in layer_info.keys() else 0
                self.nodes[layer_name] = nn.ConvTranspose2d(in_channels,
                                                            out_channels,
                                                            kernel_size,
                                                            strides,
                                                            padding,
                                                            output_padding=out_pad,
                                                            dilation=dilations)
            elif layer_type == 'ConvTranspose1d':
                kernel_size = layer_info['kernel_shape'][0]
                in_channels = layer_info['input_shape'][2]
                out_channels = layer_info['output_shape'][2]
                stride = layer_info['strides'][1]
                padding = layer_info['pads'][2]
                dilation = layer_info['dilations'][1]
                out_pad = layer_info['out_pad'][0] if 'out_pad' in layer_info.keys() else 0
                self.nodes[layer_name] = nn.ConvTranspose1d(in_channels,
                                                            out_channels,
                                                            kernel_size,
                                                            stride,
                                                            padding,
                                                            output_padding=out_pad,
                                                            dilation=dilation)
            elif layer_type == 'Pool':
                kernel_size = tuple(layer_info['kernel_shape'][1:3][::-1])
                strides = tuple(layer_info['strides'][1:3][::-1])
                padding = tuple([layer_info['pads'][4], layer_info['pads'][2]])
                pooling_type = layer_info['pooling_type']

                # if kernel_size == layer_info['kernel_shape']: # global pooling -> set kernel size accordingly
                #     kernel_size = tuple(layer_info['input_shape'][1:3][::-1])
                #     logging.debug(f'Torch: Creating layer: global {pooling_type} pool.')
                # else:
                #     logging.debug(f'Torch: Creating layer: {pooling_type} pool.')

                if pooling_type == 'MAX':
                    self.nodes[layer_name] = nn.MaxPool2d(kernel_size, strides, padding)
                elif pooling_type == 'AVG':
                    self.nodes[layer_name] = nn.AvgPool2d(kernel_size, strides, padding)
                else:
                    raise NotImplementedError("Only (global) max, average pooling implemented currently!")
            elif layer_type == 'Pool1d':
                kernel_size = layer_info['kernel_shape'][1]
                stride = layer_info['strides'][1]
                padding = layer_info['pads'][2]
                pooling_type = layer_info['pooling_type']

                # if kernel_size == -1: # global pooling -> set kernel size accordingly
                #     kernel_size = layer_info['input_shape'][1]
                #     logging.debug(f'Torch: Creating layer: global {pooling_type} 1D pool.')
                # else:
                #     logging.debug(f'Torch: Creating layer: {pooling_type} 1D pool.')

                if pooling_type == 'MAX':
                    self.nodes[layer_name] = nn.MaxPool1d(kernel_size, stride, padding)
                elif pooling_type == 'AVG':
                    self.nodes[layer_name] = nn.AvgPool1d(kernel_size, stride, padding)
                else:
                    raise NotImplementedError("Only (global) max, average pooling implemented currently!")
            elif layer_type == 'Relu':
                self.nodes[layer_name] = nn.ReLU(inplace=False)
            elif layer_type == 'Add':
                self.nodes[layer_name] = Add()
            elif layer_type in ['FullyConnected', 'MatMul']:
                self.nodes[layer_name] = nn.Linear(layer_info['input_shape'][1], layer_info['output_shape'][1])
            elif layer_type == 'Flatten':
                self.nodes[layer_name] = Flatten()
            elif layer_type == 'Softmax':
                self.nodes[layer_name] = nn.Softmax()
            elif layer_type == 'Reshape':
                print(f'Creating reshape layer with shape {layer_info}')
                self.nodes[layer_name] = Reshape(tuple(layer_info['output_shape']))
            elif layer_type == 'Squeeze':
                # make list of dimensions to squeeze
                input = layer_info['input_shape']
                output = layer_info['output_shape']
                # loop over all dimensions and check if they are squeezed
                dims_to_squeeze = []
                o = 0
                for i in range(len(input)):
                    if input[i] == output[o]:
                        o += 1
                    else:
                        dims_to_squeeze.append(i)
                self.nodes[layer_name] = Squeeze(tuple(dims_to_squeeze))
            elif layer_type == 'Relu6':
                self.nodes[layer_name] = nn.ReLU6()
            elif layer_type == 'BatchNorm':
                self.nodes[layer_name] = nn.BatchNorm2d(layer_info['input_shape'][3])
            elif layer_type == 'Concat':
                dim_map_annette_to_torch = (0, 1, 2, 3) if len(layer_info['output_shape']) == 4 else (0, 2, 1)
                annette_dim = layer_info['axis'] if 'axis' in layer_info.keys() else 1
                torch_dim = dim_map_annette_to_torch[annette_dim]
                self.nodes[layer_name] = Concat(torch_dim)
            elif layer_type == 'DataInput':
                continue # not needed in pytorch
            elif layer_type == 'Dropout':
                self.nodes[layer_name] = nn.Identity()
            elif layer_type == 'Sigmoid':
                self.nodes[layer_name] = nn.Sigmoid()
            elif layer_type == 'Mul':
                self.nodes[layer_name] = Mul()
            elif layer_type == 'Resize':
                self.nodes[layer_name] = Resize(tuple(layer_info['output_shape'][1:3][::-1]))
            else:
                raise NotImplementedError(f'Operation of type {layer_type} is not supported!')

            self.parent_nodes[layer_name] = []
            for i, l in enumerate(layer_info['parents']):
                inp_name = l if l not in annette_graph.model_spec['input_layers'] else 'inp'
                self.parent_nodes[layer_name].append(inp_name)
                    
    def forward(self, input):
        outputs = {}
        for node_name, operation in self.nodes.items():
            parents = self.parent_nodes[node_name]
            op_inputs = [outputs[o] if o != 'inp' else input for o in parents]
            outputs[node_name] = operation(*op_inputs)
        
        return tuple([outputs[o] for o in outputs if o in self.output_layers])
