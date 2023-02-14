from __future__ import print_function

import json
import logging
import pickle
import sys
from functools import reduce
from pprint import pprint

import numpy as np
import pandas as pd
from annette.estimation import layers
from annette import get_database 

__author__ = "Matthias Wess"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"


class Optimizer():
    def __init__(self, name, prim_type, sec_type, out_type, est_model=None, conv_dict=None, fuse_cond=None):
        self.name = name
        self.prim_type = prim_type
        self.sec_type = sec_type
        self.out_type = out_type
        self.est_model = est_model
        self.conv_dict = conv_dict
        self.fuse_cond = fuse_cond

        self.layer_classes = {
            'Pool': layers.PoolLayer,
            'Conv': layers.ConvLayer,
            'ConvTranspose': layers.ConvTransposeLayer,
            'ConvTranspose2d': layers.ConvTransposeLayer,
            'Add': layers.AdditionLayer,
            'Base': layers.BaseLayer,
            'DataInput': layers.InputLayer,
            'FullyConnected': layers.FullyConnectedLayer,
            'ConvPool': layers.ConvPoolLayer,
            'DepthwiseConv': layers.DepthwiseConvLayer,
            'MatMul': layers.FullyConnectedLayer,
            'DepthwiseSepConv': layers.DepthwiseSepConvLayer
        }

        self.desc = self.gen_dict()
        if self.est_model and self.conv_dict:
            self.load_model(self.est_model, self.conv_dict)

    def gen_dict(self, filename=None):
        desc = {"name": self.name,
                "prim_type": self.prim_type,
                "sec_type": self.sec_type,
                "out_type": self.out_type,
                "est_model": self.est_model,
                "conv_dict": self.conv_dict,
                "fuse_cond": self.fuse_cond}
        return desc

    def apply_split(self, graph, layer):
        # check layer type
        if layer in graph.model_spec['layers'] and graph.model_spec['layers'][layer]['type'] == self.prim_type:
            print(layer + " " + self.prim_type + " found")
            # if both match apply optimizer and generate fused layer type
            merge = True
            if merge:
                print(merge)
                self.split_simple(graph, layer)

    def split_simple(self, graph, layer):
        layertype = graph.model_spec['layers'][layer]['type']
        graph.split_layer(layer, self.out_type)
        # recompute nums
        # print(self.layer_classes[layertype].compute_nums(graph.model_spec['layers'][layer]))

    def apply_removal(self, graph, layer):
        if layer in graph.model_spec['layers'] and graph.model_spec['layers'][layer]['type'] == self.prim_type:
            logging.debug(f'Removing layer: {layer}')
            graph.delete_layer(layer)

    def apply_merge(self, graph, layer):
        # check layer type
        if layer in graph.model_spec['layers'] and graph.model_spec['layers'][layer]['type'] == self.prim_type:
            print(layer + " " + self.prim_type + " found")
            # check how many children
            if len(graph.model_spec['layers'][layer]['children']) == 1:
                c = graph.model_spec['layers'][layer]['children'][0]
                # check output layer type
                print(graph.model_spec['layers'][c]['type'], self.sec_type)
                if graph.model_spec['layers'][c]['type'] == self.sec_type:
                    # if both match apply optimizer and generate fused layer type
                    merge = True
                    if merge and self.fuse_cond:
                        print("Conditional Merge")
                        print(merge)
                        print(self.fuse_cond)
                        merge = merge and self.merge_cond(graph, layer)
                    if merge and self.est_model:
                        print("Model based Optimizer")
                        print(merge)
                        print(self.est_model)
                        print(self.conv_dict)
                        merge = merge and self.merge_model(graph, layer)
                    print("Merge Decision")
                    if merge:
                        print(merge)
                        self.merge_simple(graph, layer)

    def merge_simple(self, graph, layer):
        layertype = graph.model_spec['layers'][layer]['type']
        c = graph.model_spec['layers'][layer]['children'][0]
        # graph.delete_layer(c)
        graph.fuse_layer(layer, c)
        graph.model_spec['layers'][layer]['type'] = self.out_type
        #print(self.layer_classes[layertype].compute_nums(graph.model_spec['layers'][layer]))

    def merge_cond(self, graph, layer):
        print("conditional merge")
        p_layertype = graph.model_spec['layers'][layer]['type']
        c = graph.model_spec['layers'][layer]['children'][0]
        try:
            primary = self.layer_classes[p_layertype].compute_nums(
                graph.model_spec['layers'][layer])
        except:
            primary = graph.model_spec['layers'][layer]

        s_layertype = graph.model_spec['layers'][c]['type']
        try:
            secondary = self.layer_classes[s_layertype].compute_nums(
                graph.model_spec['layers'][c])
        except:
            secondary = graph.model_spec['layers'][c]

        print(primary, secondary)
        print(self.fuse_cond)

        def cond_check(cond, val, verb=None):
            if cond['cond'] == "==" and str(val) == cond['val']:
                res = 0
            elif cond['cond'] == ">" and val > float(cond['val']):
                res = 0
            elif cond['cond'] == "<" and val < float(cond['val']):
                res = 0
            elif cond['cond'] == ">=" and val >= float(cond['val']):
                res = 0
            elif cond['cond'] == "<=" and val <= float(cond['val']):
                res = 0
            else:
                res = 1

            if verb and res == 1:
                print(cond['val']+cond['cond']+str(val)+" not fulfilled")
            elif verb and res == 0:
                print(cond['val']+cond['cond']+str(val)+" fulfilled")
            return res

        count_cond = 0
        # check primary conditions
        for key, cond in self.fuse_cond['primary'].items():
            # print(cond)
            if 'i' in cond:
                val = primary[cond['name']][cond['i']]
            else:
                val = primary[cond['name']]
            count_cond += cond_check(cond, val)

        # check secondary conditions
        for key, cond in self.fuse_cond['secondary'].items():
            # print(cond)
            if 'i' in cond:
                val = secondary[cond['name']][cond['i']]
            else:
                val = secondary[cond['name']]
            count_cond += cond_check(cond, val)

        logging.debug(count_cond)

        if count_cond == 0:
            return True
        else:
            return False

    def merge_model(self, graph, layer):
        print("merge model")
        print("statistical estimation")

        p_layertype = graph.model_spec['layers'][layer]['type']
        c = graph.model_spec['layers'][layer]['children'][0]
        try:
            primary = self.layer_classes[p_layertype].compute_nums(
                graph.model_spec['layers'][layer])
        except:
            primary = graph.model_spec['layers'][layer]

        s_layertype = graph.model_spec['layers'][c]['type']
        try:
            secondary = self.layer_classes[s_layertype].compute_nums(
                graph.model_spec['layers'][c])
        except:
            secondary = graph.model_spec['layers'][c]

        vector = np.zeros([1, len(self.conv_dict)])
        for i in self.conv_dict.items():
            if isinstance(i[1], dict):
                if i[1]['layer'] == "secondary":
                    choser = secondary
                else:
                    choser = primary
                if 'i' in i[1]:
                    vector[0, int(i[0])] = choser[i[1]['name']][i[1]['i']]
                else:
                    vector[0, int(i[0])] = choser[i[1]['name']]
                if 'dec' in i[1].keys():
                    vector[0, i[0]] = vector[0, i[0]] - i[1]['dec']
            else:
                vector[0, int(i[0])] = i[1]
        result = self.est_model.predict(vector)
        print("estimation result")
        print(result)
        if result == 1.0:
            return True
        else:
            return False

    def load_model(self, est_model=None, conv_dict=None):
        if est_model != None:
            self.est_model = pickle.load(open(get_database(est_model), 'rb'))
            self.desc['est_model'] = est_model
        else:
            print("No Model to be loaded")
            pass

        self.conv_dict = conv_dict
        self.desc['conv_dict'] = conv_dict
        print("Conversion Dict: ")
        print(self.conv_dict)


class Mapping_model():
    def __init__(self, name):
        self.name = name
        self.optimizers = {}

    def add_optimizer(self, name, prim_type, sec_type, out_type, est_model=None, conv_dict=None, fuse_cond=None):
        self.optimizers[name] = Optimizer(
            name, prim_type, sec_type, out_type, est_model, conv_dict, fuse_cond)

    def run_optimization(self, graph):
        print(graph.topological_sort)
        # loop over optimizers
        print("Optimizers:")
        for key, opt in self.optimizers.items():
            print(key)
            # loop over layers
            if opt.sec_type is None:
                if opt.out_type is None:
                    for layer in graph.topological_sort:
                        opt.apply_removal(graph, layer)
                else:
                    print("Splitter")
                    for layer in graph.topological_sort:
                        opt.apply_split(graph, layer)
        for key, opt in self.optimizers.items():
            if opt.sec_type is not None:
                print("Fuser")
                for layer in graph.topological_sort:
                    opt.apply_merge(graph, layer)
        print(graph.topological_sort)

    def to_json(self, filename=None):
        """Store Mapping Estimator to json file"""
        mapping_desc = {}
        mapping_desc['name'] = self.name
        for o in self.optimizers.items():
            mapping_desc[o[0]] = o[1].desc
            print(o)
            print(o[1].desc)
        if filename:
            with open(filename, 'w') as f:
                f.write(json.dumps(mapping_desc, indent=4))

    @classmethod
    def from_json(cls, filename):
        """Reconstruct Mapping Estimator from json file"""

        with open(filename, 'r') as f:
            opt_desc = json.load(f)

        output_model = cls(opt_desc['name'])

        print("Initial % d entries..." % len(opt_desc))
        del opt_desc['name']
        print(len(opt_desc))
        for l in opt_desc.items():
            print(l[0])
            print(l[1])
            name = l[1]['name']
            prim_type = l[1]['prim_type']
            sec_type = l[1]['sec_type']
            out_type = l[1]['out_type']
            opts = {}
            opts_list = ['est_model', 'conv_dict', 'fuse_cond']
            for i in opts_list:
                if i in l[1]:
                    opts[i] = l[1][i]
                else:
                    opts[i] = None

            output_model.add_optimizer(name, prim_type, sec_type, out_type,
                                       est_model=opts['est_model'], conv_dict=opts['conv_dict'], fuse_cond=opts['fuse_cond'])

        return output_model


def main():
    print("load")
