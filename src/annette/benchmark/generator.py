from __future__ import print_function
from annette.estimation import layers
import json
import numpy as np
import pandas as pd
import pickle as pkl
import logging
from pathlib import Path
import tensorflow.compat.v1 as tf
#import tensorflow.compat.v1.contrib.slim as slim
import os
from copy import deepcopy
import subprocess

from annette import get_database 
from annette.graph import AnnetteGraph

#TODO renaming of tf_variables and move some things to the tf file

def generate_tf_model(graph):
    """generates Tensorflow 2 graph out of ANNETTE graph description
    and stores to benchmark/graphs/tf2/ directory

    Args:
        graph :obj:`annette.graph.AnnetteGraph`: annette graph description to generate the tf2 graph from
    """

    # generate tensorflow model and export to out_file

    # with __dict__ we can see the content of the class
    logging.debug(graph.__dict__)

    # model_spec contains some info about the model
    for key, value  in graph.model_spec.items():
        logging.debug(key)
        logging.debug(value)

    network_name = graph.model_spec['name']

    filename = get_database( 'benchmark', 'graphs' ,'tf2', network_name+'.pb')
    logging.debug("Stored to: %s" % filename)


class Graph_generator():
    """Graph generator"""

    def __init__(self, network):
        #load graphstruct
        self.json_file = get_database('graphs','annette',network+'.json')
        self.init_graph = AnnetteGraph(network, self.json_file)
        self.graph = deepcopy(self.init_graph)
        #load configfile
        self.config = None
    
    def add_configfile(self, configfile):
        self.config = pd.read_csv(get_database('benchmarks','config', configfile))
        print(self.config)

    def generate_graph_from_config(self, num):
        # can be used as to generate input for generate_tf_model
        # execute the function under test
        self.graph = deepcopy(self.init_graph)

        def replace_key(value, config, num):
            if value in self.config.keys().to_numpy():
                logging.debug("%s detected", value)
                return int(self.config.iloc[num][value])
            else:
                return value

        # model_spec contains some info about the model
        for key, value  in self.graph.model_spec.items():
            logging.debug(key)
            logging.debug(value)

        if tf.__version__[0] == '2':
            tf.compat.v1.reset_default_graph()
        else:
            tf.reset_default_graph()
        self.tf_graph = {}

        if self.config:
            for layer_n, layer_attrs in self.graph.model_spec['layers'].items():
                logging.debug("layer name %s " % layer_n)
                logging.debug("layer attrs %s " % layer_attrs)
                for attr_n,attr_v in layer_attrs.items():
                    logging.debug("attribute name %s" % attr_n)
                    logging.debug("attribute values %s" % attr_v)

                    if isinstance(attr_v, list):
                        for n,attr_ele in enumerate(attr_v):
                            #logging.debug(n)
                            #logging.debug(attr_ele)
                            self.graph.model_spec['layers'][layer_n][attr_n][n] = replace_key(attr_ele, self.config, num)
                    else:
                        self.graph.model_spec['layers'][layer_n][attr_n] = replace_key(attr_v, self.config, num)

        self.graph.compute_dims()

        logging.debug("Loop through layers")

        for layer_n, layer_attrs in self.graph.model_spec['layers'].items():
            if layer_attrs['type'] == "DataInput":
                self.tf_graph[layer_n] = self.tf_gen_placeholder(layer_attrs, layer_n)
            elif layer_attrs['type'] == "Conv":
                self.tf_graph[layer_n] = self.tf_gen_conv(layer_attrs, layer_n)
            elif layer_attrs['type'] == "Relu":
                self.tf_graph[layer_n] = self.tf_gen_relu(layer_attrs, layer_n)
            elif layer_attrs['type'] == "Add":
                self.tf_graph[layer_n] = self.tf_gen_add(layer_attrs, layer_n)
            elif layer_attrs['type'] == "DepthwiseConv":
                self.tf_graph[layer_n] = self.tf_gen_dwconv(layer_attrs, layer_n)
            elif layer_attrs['type'] == "Pool":
                self.tf_graph[layer_n] = self.tf_gen_pool(layer_attrs, layer_n)
            elif layer_attrs['type'] == "Concat":
                self.tf_graph[layer_n] = self.tf_gen_concat(layer_attrs, layer_n)
            elif layer_attrs['type'] == "Flatten":
                self.tf_graph[layer_n] = self.tf_gen_flatten(layer_attrs, layer_n)
            elif layer_attrs['type'] == "Squeeze":
                self.tf_graph[layer_n] = self.tf_gen_flatten(layer_attrs, layer_n)
            elif layer_attrs['type'] == "Softmax":
                self.tf_graph[layer_n] = self.tf_gen_softmax(layer_attrs, layer_n)
            elif layer_attrs['type'] == "MatMul" or layer_attrs['type'] == "FullyConnected": # TODO check this! Maybe FullyConnected with bias
                self.tf_graph[layer_n] = self.tf_gen_matmul(layer_attrs, layer_n)
            elif layer_attrs['type'] == "Relu6":
                self.tf_graph[layer_n] = self.tf_gen_relu6(layer_attrs, layer_n)
            elif layer_attrs['type'] == "BatchNorm":
                self.tf_graph[layer_n] = self.tf_gen_batchnorm(layer_attrs, layer_n)
            elif layer_attrs['type'] == "Reshape":
                self.tf_graph[layer_n] = self.tf_gen_reshape(layer_attrs, layer_n)
            else:
                logging.debug("layer %s not yet implemented", layer_attrs['type'])
                exit()

            if self.config:
                logging.debug("Config %s" % self.config.iloc[num])
            logging.debug("Current graph %s" % self.tf_graph)

        # return annette graph
        out = deepcopy(self.graph.model_spec['output_layers'])
        logging.debug(self.graph.model_spec)

        for i, o in enumerate(out):
            print(i,o)
            if self.graph.model_spec['layers'][o]['type'] == 'BatchNorm':
                out[i] = o+'/add'
        self.tf_export_to_pb(out)
        return out 

    def lite_to_onnx(self, input_nodes, output_nodes, input_shapes, load_path= None, save_path = None):
        # Convert the model.
        subprocess.call("python -m tf2onnx.convert --opset 13 --tflite "+str(load_path)+" --output "+str(save_path), shell=True)

    def tf2_export_to_lite(self, input_nodes, output_nodes, input_shapes, load_path= None, save_path = None):
        # Convert the model.
        converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
            graph_def_file= load_path,
                            # both `.pb` and `.pbtxt` files are accepted.
            input_arrays= input_nodes,
            input_shapes= input_shapes,
            output_arrays=output_nodes
        )
        #tf.contrib.quantize.create_eval_graph()
        #converter.optimizations = [tf.lite.Optimize.DEFAULT]
        #converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(model_fn)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # Set inputs and outputs of network to 8-bit unsigned integer
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

        def representative_dataset_gen():
            for _ in range(250):
                yield [np.random.uniform(0.0, 0.0, size=input_shapes['input']).astype(np.float32)]
        converter.representative_dataset = representative_dataset_gen
        converter._experimental_new_quantizer = True

        tflite_model = converter.convert()

        # Save thesave_path
        with open(save_path, 'wb') as f:
            f.write(tflite_model) 

    def tf_export_to_l(self, input_nodes, output_nodes, input_shapes, load_path= None, save_path = None):
        # Collect default graph information
        g = tf.get_default_graph()

        with tf.Session() as sess:
            # Initialize the variables
            sess.run(tf.global_variables_initializer())
            g = g.as_graph_def(add_shapes = True)
            tf.contrib.quantize.create_eval_graph()

            # Convert variables to constants until the "fully_conn_1/Softmax" node
            frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, g, output_nodes)

            print("load graph")
            graph_nodes=[n for n in frozen_graph_def.node]
            names = []
            for t in graph_nodes:
                if not ("Variable" in t.name or "BiasAdd" in t.name):
                    names.append(t.name.replace("/","_").replace("-","_"))
        converter = tf.lite.TFLiteConverter.from_frozen_graph(
            graph_def_file= load_path,
                            # both `.pb` and `.pbtxt` files are accepted.
            input_arrays= input_nodes,
            input_shapes= input_shapes,
            output_arrays=output_nodes
        )

        tf.contrib.quantize.create_eval_graph()
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.default_ranges_stats = (0,1)


        tflite_model = converter.convert()

        # Save thesave_path
        with open(save_path, 'wb') as f:
            f.write(tflite_model) 

        exit()

    def tf_export_to_pb(self, output_node, save_path = None):
        # Collect default graph information
        g = tf.get_default_graph()

        with tf.Session() as sess:
            # Initialize the variables
            sess.run(tf.global_variables_initializer())
            g = g.as_graph_def(add_shapes = True)

            # Convert variables to constants until the "fully_conn_1/Softmax" node
            frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, g, output_node)

            print("load graph")
            graph_nodes=[n for n in frozen_graph_def.node]
            names = []
            for t in graph_nodes:
                if not ("Variable" in t.name or "BiasAdd" in t.name):
                    names.append(t.name.replace("/","_").replace("-","_"))

        # Write the intermediate representation of the graph to .pb file
        if save_path:
            net_file = save_path
        else:
            net_file = get_database('graphs','tf',self.graph.model_spec['name']+".pb")
        #print(net_file)
        with open(os.path.join(net_file), 'wb') as f:
            graph_string = (frozen_graph_def.SerializeToString())
            f.write(graph_string)

    def tf2_export_to_pb(self, output_node, save_path = None):
        # Collect default graph information
        g = tf.compat.v1.get_default_graph()

        with tf.compat.v1.Session() as sess:
            # Initialize the variables
            sess.run(tf.compat.v1.global_variables_initializer())
            g = g.as_graph_def(add_shapes = True)

            # Convert variables to constants until the "fully_conn_1/Softmax" node
            frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess, g, output_node)

            print("load graph")
            graph_nodes=[n for n in frozen_graph_def.node]
            names = []
            for t in graph_nodes:
                if not ("Variable" in t.name or "BiasAdd" in t.name):
                    names.append(t.name.replace("/","_").replace("-","_"))

        # Write the intermediate representation of the graph to .pb file
        if save_path:
            net_file = save_path
        else:
            net_file = get_database('graphs','tf',self.graph.model_spec['name']+".pb")
        #print(net_file)
        with open(os.path.join(net_file), 'wb') as f:
            graph_string = (frozen_graph_def.SerializeToString())
            f.write(graph_string)

    def tf_gen_pool(self, layer, name=None):
        logging.debug("Generating Pool with dict: %s" % layer)
        inp_name = layer['parents'][0]
        inp = self.tf_graph[inp_name]
        k_w = layer['kernel_shape'][1]
        k_h = layer['kernel_shape'][2]
        stride_w = layer['strides'][1]
        stride_h = layer['strides'][2]
        pad = 'SAME'
        if sum(layer['pads']) == 0:
            pad = 'VALID'
        if layer['pooling_type'] == 'MAX':
            return maxpool(inp, (k_w, k_h),(stride_w, stride_h), name)
        elif layer['pooling_type'] == 'AVG' and layer['kernel_shape'][1] == -1:
            return globavgpool(inp, name)
        elif layer['pooling_type'] == 'AVG':
            return avgpool(inp, (k_w, k_h),(stride_w, stride_h), pad, name)
        else:
            logging.error("not all pooling options implementes up to now")
            exit()

    def tf_gen_concat(self, layer, name=None):
        logging.debug("Generating Concat with dict: %s" % layer)
        inp_name0 = layer['parents'][0]
        inp_name1 = layer['parents'][1]
        inp = [self.tf_graph[x] for x in layer['parents']]
        return tf.concat(inp,axis=3,name=name)

    def tf_gen_add(self, layer, name=None):
        logging.debug("Generating Add with dict: %s" % layer)
        if len(layer['parents']) == 2:
            inp_name0 = layer['parents'][0]
            inp_name1 = layer['parents'][1]
            inp0 = self.tf_graph[inp_name0]
            inp1 = self.tf_graph[inp_name1]
            return tf.add(inp0, inp1, name=name)
        else:
            raise NotImplementedError

    def tf_gen_flatten(self, layer, name=None):
        logging.debug("Generating Flatten with dict: %s" % layer)
        inp_name = layer['parents'][0]
        inp = self.tf_graph[inp_name]
        return flatten(inp, name)

    def tf_gen_reshape(self, layer, name=None):
        logging.debug("Generating Reshape with dict: %s" % layer)
        inp_name = layer['parents'][0]
        inp = self.tf_graph[inp_name]
        print(layer['output_shape'])
        return reshape(inp, layer['output_shape'], name)

    def tf_gen_batchnorm(self, layer, name=None):
        logging.debug("Generating Batchnorm with dict: %s" % layer)
        inp_name = layer['parents'][0]
        inp = self.tf_graph[inp_name]
        return batch_norm(inp, name)

    def tf_gen_relu(self, layer, name=None):
        logging.debug("Generating Relu with dict: %s" % layer)
        inp_name = layer['parents'][0]
        inp = self.tf_graph[inp_name]
        return relu(inp, name)

    def tf_gen_relu6(self, layer, name=None):
        logging.debug("Generating Relu6 with dict: %s" % layer)
        inp_name = layer['parents'][0]
        inp = self.tf_graph[inp_name]
        return relu6(inp, name)

    def tf_gen_softmax(self, layer, name=None):
        logging.debug("Generating Softmax with dict: %s" % layer)
        inp_name = layer['parents'][0]
        inp = self.tf_graph[inp_name]
        return softmax(inp, name)

    def tf_gen_matmul(self, layer, name=None):
        logging.debug("Generating MatMul with dict: %s" % layer)
        inp_name = layer['parents'][0]
        inp = self.tf_graph[inp_name]
        filters = layer['output_shape'][1]
        return matmul(inp, filters, name)

    def tf_gen_conv(self, layer, name=None):
        logging.debug("Generating Conv with dict: %s" % layer)
        inp_name = layer['parents'][0]
        inp = self.tf_graph[inp_name]
        filters = layer['output_shape'][3]
        k_w = layer['kernel_shape'][0]
        k_h = layer['kernel_shape'][1]
        stride_w = layer['strides'][1]
        stride_h = layer['strides'][2]
        return conv2d(inp, filters, (k_w,k_h), (stride_w,stride_h), name)

    def tf_gen_dwconv(self, layer, name=None):
        logging.debug("Generating DWConv with dict: %s" % layer)
        inp_name = layer['parents'][0]
        k_w = layer['kernel_shape'][0]
        k_h = layer['kernel_shape'][1]
        s_w = layer['strides'][1]
        s_h = layer['strides'][2]
        inp = self.tf_graph[inp_name]
        filters = layer['output_shape'][3]
        #return tf.layers.separable_conv2d(inp, filters, (k_w,k_h), padding='same')
        return dw_conv2d(inp, (k_w,k_h), (s_w, s_h), name)

    def tf_gen_placeholder(self, layer, name="x"):
        logging.debug("Generating Placeholder with dict: %s" % layer)
        batch_size = layer['output_shape'][0]
        if batch_size == -1:
            batch_size = 1
        width = layer['output_shape'][1] 
        height = layer['output_shape'][2]
        channels = layer['output_shape'][3]
        tf.compat.v1.disable_eager_execution()
        return tf.compat.v1.placeholder(tf.float32, [batch_size, width, height, channels], name=name)

def dw_conv2d(x_tensor, conv_ksize, stride, name):
    
    #layer = slim.separable_convolution2d(x_tensor,
    #                                        num_outputs=None,
    #                                        stride=stride,
    #                                        depth_multiplier=1,
    #                                        kernel_size=conv_ksize,
    #                                        scope=name,
    #                                        activation_fn=None)
                                            
    strides = [1] + list(stride) + [1]
    W_shape = list(conv_ksize) + [int(x_tensor.shape[3]), 1]
    W = tf.Variable(tf.random.truncated_normal(W_shape, stddev=.05))
    layer = tf.raw_ops.DepthwiseConv2dNative(input=x_tensor, filter=W, strides=strides, padding="SAME", name=name)


    #layer = tf.keras.layers.Conv2D(int(x_tensor.shape[3]), list(conv_ksize), padding='valid', groups=int(x_tensor.shape[3]), use_bias=False)

    #layer = tf.layers.separable_conv2d(x_tensor,
    #                                        strides=stride,
    #                                        depth_multiplier=1,
    #                                        kernel_size=conv_ksize,
    #                                        name=name)
    return layer
    

def conv2d(x_tensor, filters, conv_ksize, stride, name):
    # Weights
    conv_strides = stride
    W_shape = list(conv_ksize) + [int(x_tensor.shape[3]), filters]
    W = tf.Variable(tf.random.truncated_normal(W_shape, stddev=.05))

    # Apply convolution
    x = tf.nn.conv2d(
        x_tensor, W,
        strides = [1] + list(conv_strides) + [1],
        padding = 'SAME',
        name = name
    )

    return x

def relu(x_tensor, name):
    # Nonlinear activation (ReLU)
    x = tf.nn.relu(x_tensor,name=name)
    return x

def relu6(x_tensor, name):
    # Nonlinear activation (ReLU6)
    x = tf.nn.relu6(x_tensor,name=name)
    return x

def batch_norm(x_tensor, name):
    # batch normalization
    x = tf.nn.batch_normalization(x_tensor,0,0,0,1,1e-5,name=name)
    return x

def softmax(x_tensor, name):
    # Nonlinear activation (Softmax)
    x = tf.nn.softmax(x_tensor,name=name)
    return x

def globavgpool(x_tensor, name='avg_pool'):
    x = tf.reduce_mean(x_tensor, axis=[1,2], name = name)
    return x

def maxpool(x_tensor, pool_ksize, pool_strides, name='max_pool'):
    # Max pooling
    x = tf.nn.max_pool(
        x_tensor,
        ksize = [1] + list(pool_ksize) + [1],
        strides = [1] + list(pool_strides) + [1],
        padding = 'SAME',
        name = name
    )
    return x

def avgpool(x_tensor, pool_ksize, pool_strides, pads='SAME', name='avg_pool'):
    x = tf.nn.avg_pool(
            x_tensor,
            ksize = [1] + list(pool_ksize) + [1],
            strides = [1] + list(pool_strides) + [1],
            padding = pads,
            name = name
        )

    return x

def flatten(x_tensor, name):
    x = tf.reshape(x_tensor, [1, np.prod(x_tensor.shape.as_list()[1:])], name = name)
    return x

def reshape(x_tensor, reshape, name):
    x = tf.reshape(x_tensor, reshape, name = name)
    return x

def matmul(x_tensor, num_outputs, name):
    # Weights and bias
    s = [int(x_tensor.shape[1]), num_outputs]
    W = tf.Variable(tf.random.truncated_normal(s , stddev=.05))
    # The fully connected layer
    x = tf.matmul(x_tensor, W, name=name)
    return x

def output(x_tensor, num_outputs):
    with tf.name_scope('fully_conn'):
        # Weights and bias
        W = tf.Variable(tf.random.truncated_normal([int(x_tensor.shape[1]), num_outputs], stddev=.05))
        b = tf.Variable(tf.ones([num_outputs]))

        # The output layer
        x = tf.add(tf.matmul(x_tensor, W), b)
        x = tf.nn.softmax(x)
    return x
