from __future__ import print_function
from annette.estimation import layers
import json
import numpy as np
import pandas as pd
import pickle as pkl
import logging
from pathlib import Path
import tensorflow as tf
import os

from annette import get_database 
from annette.graph import AnnetteGraph

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

"""
        width = int(df.at[index, 'width'])
        height = int(df.at[index, 'height'])
        channels = int(df.at[index, 'channels'])
        filters = int(df.at[index, 'filters'])
        kernel_size = int(df.at[index, 'k_size'])
        batch_size = 1
        stride = 1

        # Remove previous weights, bias, inputs, etc..
        tf.compat.v1.reset_default_graph()

        # Inputs in format: [b_s,h,w,channels]
        x = tf.compat.v1.placeholder(tf.float32, [batch_size, width, height, channels], name="x")

        output_node = "" # Name of the model output node, differs for different networks
        k = (kernel_size, kernel_size)
        maxpool_param = (2,2)

#First Block
        print(k)

        xq = conv2d(x, filters, k, stride)
        xq_1 = conv2d(x, filters, (kernel_size,1), stride)
        x1_q = conv2d(x, filters, (1,kernel_size), stride)
        x = tf.add(xq,xq_1)
        x = tf.add(x,x1_q)

        x = tf.layers.separable_conv2d(x, channels, k, padding='same')

        x1 = conv2d(x, filters, k, 2)

        x = conv2d(x, filters, k, 1)
        x = maxpool(x, maxpool_param, maxpool_param)

        x = tf.concat([x1,x],axis=3)

        output_node = "concat"
"""    

class Graph_generator():
    """Bench generator"""

    def __init__(self, network):
        #load graphstruct
        json_file = get_database('graphs','annette',network+'.json')
        self.graph = AnnetteGraph(network, json_file)
        print(self.graph)
        #load configfile
    
    def add_configfile(self, configfile):
        self.config = pd.read_csv(get_database('benchmarks','config', configfile))
        print(self.config)

    def generate_graph_from_config(self, num):
        # can be used as to generate input for generate_tf_model
        # execute the function under test

        def replace_key(value, config, num):
            if value in self.config.keys():
                logging.debug("%s detected", value)
                return int(self.config.iloc[num][value])
            else:
                return value

        # model_spec contains some info about the model
        for key, value  in self.graph.model_spec.items():
            logging.debug(key)
            logging.debug(value)

        tf.compat.v1.reset_default_graph()
        self.tf_graph = {}

        logging.debug("Loop through layers")
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
            else:
                print("no layer")
                exit()

            logging.debug("Config %s" % self.config.iloc[num])
            logging.debug("Current graph %s" % self.tf_graph)

        # return annette graph
        self.tf_export_to_pb("Add_1")
        return None

    def tf_export_to_pb(self, output_node):
        # Collect default graph information
        g = tf.get_default_graph()

        with tf.Session() as sess:
            # Initialize the variables
            sess.run(tf.global_variables_initializer())
            g = g.as_graph_def(add_shapes = True)

            # Convert variables to constants until the "fully_conn_1/Softmax" node
            frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, g, [output_node])

            print("load graph")
            graph_nodes=[n for n in frozen_graph_def.node]
            names = []
            for t in graph_nodes:
                if not ("Variable" in t.name or "BiasAdd" in t.name):
                    names.append(t.name.replace("/","_").replace("-","_"))
            print(names)

        # Write the intermediate representation of the graph to .pb file
        with open(os.path.join("network.pb"), 'wb') as f:
            graph_string = (frozen_graph_def.SerializeToString())
            f.write(graph_string)

    def tf_gen_pool(self, layer, name=None):
        logging.debug("Generating Relu with dict: %s" % layer)
        inp_name = layer['parents'][0]
        inp = self.tf_graph[inp_name]
        k_w = layer['kernel_shape'][2]
        k_h = layer['kernel_shape'][3]
        stride_w = layer['strides'][1]
        stride_h = layer['strides'][2]
        if layer['pooling_type'] != 'MAX':
            logging.error("only max pooling implemented currently")
            exit()
        
        return maxpool(inp, (k_w, k_h),(stride_w, stride_h), name)

    def tf_gen_concat(self, layer, name=None):
        logging.debug("Generating Add with dict: %s" % layer)
        inp_name0 = layer['parents'][0]
        inp_name1 = layer['parents'][1]
        inp0 = self.tf_graph[inp_name0]
        inp1 = self.tf_graph[inp_name1]
        return tf.add(inp0, inp1)

    def tf_gen_add(self, layer, name=None):
        logging.debug("Generating Add with dict: %s" % layer)
        inp_name0 = layer['parents'][0]
        inp_name1 = layer['parents'][1]
        inp0 = self.tf_graph[inp_name0]
        inp1 = self.tf_graph[inp_name1]
        return tf.add(inp0, inp1, name=name)

    def tf_gen_relu(self, layer, name=None):
        logging.debug("Generating Relu with dict: %s" % layer)
        inp_name = layer['parents'][0]
        filters = layer['output_shape'][3]
        inp = self.tf_graph[inp_name]
        return relu(inp, name)

    def tf_gen_conv(self, layer, name=None):
        logging.debug("Generating Conv with dict: %s" % layer)
        inp_name = layer['parents'][0]
        inp = self.tf_graph[inp_name]
        filters = layer['output_shape'][3]
        k_w = layer['kernel_shape'][2]
        k_h = layer['kernel_shape'][3]
        stride_w = layer['strides'][1]
        stride_h = layer['strides'][2]
        return conv2d(inp, filters, (k_w,k_h), (stride_w,stride_h), name)

    def tf_gen_dwconv(self, layer, name=None):
        logging.debug("Generating DWConv with dict: %s" % layer)
        inp_name = layer['parents'][0]
        k_w = layer['kernel_shape'][2]
        k_h = layer['kernel_shape'][3]
        inp = self.tf_graph[inp_name]
        filters = layer['output_shape'][3]
        return tf.layers.separable_conv2d(inp, filters, (k_w,k_h), padding='same')

    def tf_gen_placeholder(self, layer, name="x"):
        logging.debug("Generating Placeholder with dict: %s" % layer)
        batch_size = layer['output_shape'][0]
        width = layer['output_shape'][1] 
        height = layer['output_shape'][2]
        channels = layer['output_shape'][3]
        return tf.compat.v1.placeholder(tf.float32, [batch_size, width, height, channels], name=name)


def conv2d(x_tensor, filters, conv_ksize, stride, name):
    # Weights
    conv_strides = stride
    W_shape = list(conv_ksize) + [int(x_tensor.shape[3]), filters]
    W = tf.Variable(tf.truncated_normal(W_shape, stddev=.05))

    # Apply convolution
    x = tf.nn.conv2d(
        x_tensor, W,
        strides = [1] + list(conv_strides) + [1],
        padding = 'SAME',
        name = name
    )
    # Add bias
    b = tf.Variable(tf.zeros([filters]))
    x = tf.nn.bias_add(x, b)

    return x

def relu(x_tensor, name):
    # Nonlinear activation (ReLU)
    x = tf.nn.relu(x_tensor,name=name)
    return x


def maxpool(x_tensor, pool_ksize, pool_strides, name='max_pool'):
    with tf.name_scope(name):
        # Max pooling
        x = tf.nn.max_pool(
            x_tensor,
            ksize = [1] + list(pool_ksize) + [1],
            strides = [1] + list(pool_strides) + [1],
            padding = 'SAME'
        )
    return x


def flatten(x_tensor):
    with tf.name_scope('flatten'):
        x = tf.reshape(x_tensor, [-1, np.prod(x_tensor.shape.as_list()[1:])])
    return x


def fully_conn(x_tensor, num_outputs):
    with tf.name_scope('fully_conn'):
        # Weights and bias
        W = tf.Variable(tf.truncated_normal([int(x_tensor.shape[1]), num_outputs], stddev=.05))
        b = tf.Variable(tf.zeros([num_outputs]))
        # The fully connected layer
        x = tf.add(tf.matmul(x_tensor, W), b)

        # Nonlinear activation (ReLU)
        x = tf.nn.relu(x)
    return x


def output(x_tensor, num_outputs):
    with tf.name_scope('fully_conn'):
        # Weights and bias
        W = tf.Variable(tf.truncated_normal([int(x_tensor.shape[1]), num_outputs], stddev=.05))
        b = tf.Variable(tf.zeros([num_outputs]))

        # The output layer
        x = tf.add(tf.matmul(x_tensor, W), b)
        x = tf.nn.softmax(x)
    return x

