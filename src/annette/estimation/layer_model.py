from __future__ import print_function
from pprint import pprint
from functools import reduce
import numpy as np
from annette.estimation import layers
import json
import pandas as pd
import logging

class Layer_model():
    def __init__(self, name, op_s, bandwidth, architecture = None):
        self.name = name
        self.op_s = op_s 
        self.bandwidth = bandwidth
        if architecture:
            self.architecture = architecture 

        #Introduced Layer classes #TODO make this editable (not clean yet)
        self.layer_classes = {
            'Pool': layers.PoolLayer,
            'Conv': layers.ConvLayer,
            'ConvTranspose': layers.ConvTransposeLayer,
            'Add' : layers.AdditionLayer,
            'Base': layers.BaseLayer,
            'Input': layers.InputLayer,
            'FullyConnected': layers.FullyConnectedLayer,
            'ConvPool': layers.ConvPoolLayer,
            'DepthwiseConv': layers.DepthwiseConvLayer,
            }
        
        self.layer_dict = {}
        print(self.layer_dict)
    
    def add_layer(self, name, layer_type, est_type, op_s, bandwidth, architecture = None, est_model = None, conv_dict = None, y_val = 'ops/s'):
        #check if architecture available
        if est_type in ["mixed","refined_roofline"]:
            if architecture is None:
                logging.error("could not build layer because no architecture defined")
                pass
        if est_type in ["mixed", "statistical"]:
            #check if conv_dict available
            if (conv_dict is None) or (est_model is None):
                logging.error("could not build layer because no conversion dictionary available")
        
        #check which type of layer
        if layer_type in self.layer_classes:
            #generate layer
            print("generate {}-layer".format(layer_type))
            #add to layer_dict
            self.layer_dict[layer_type] = self.layer_classes[layer_type](name,layer_type,est_type,op_s,bandwidth,architecture)

            #check which type of estimation
            if est_type in ["mixed", "statistical"]:
                print("load statistical estimator: {}...".format(est_model))
                print("dictionary: {}...".format(conv_dict))
                self.layer_dict[layer_type].load_estimator(est_model, conv_dict)
            
            self.layer_dict[layer_type].y_val = y_val 
        else:
            #generate layer
            print("generate {}-layer".format(layer_type))
            #add to layer_dict
            self.layer_dict[layer_type] = self.layer_classes['Base'](name,layer_type,est_type,op_s,bandwidth,architecture)

            #check which type of estimation
            if est_type in ["mixed", "statistical"]:
                print("load statistical estimator: {}...".format(est_model))
                print("dictionary: {}...".format(conv_dict))
                self.layer_dict[layer_type].load_estimator(est_model, conv_dict)
            
            self.layer_dict[layer_type].y_val = y_val 

        pass


    def estimate_model(self, model):
        result = {}
        sum_result = 0
        #print(model.model_spec['layers'])
        result_pd = pd.DataFrame({"name":[],"type":[],"time(ms)":[]})
        result_pd['num_ops'] = np.nan
        result_pd['num_inputs'] = np.nan
        result_pd['num_outputs'] = np.nan
        result_pd['num_weights'] = np.nan
        #Add info to layer stuff
        """Loop through Layers"""
        #for layer_name, layer_info in model.model_spec['layers'].items(): 
        for layer_name in model.topological_sort: 
            layer_info = model.model_spec['layers'][layer_name]
            layer_time = 0

            #TODO Again not clean yet
            if layer_info['type'] == 'Conv' and 'Conv' in self.layer_dict:
                layer_time = self.layer_dict['Conv'].estimate(layer_info)
            elif layer_info['type'] == 'DepthwiseConv' and 'DepthwiseConv' in self.layer_dict:
                layer_time = self.layer_dict['DepthwiseConv'].estimate(layer_info)
            elif layer_info['type'] == 'ConvTranspose' and 'ConvTranspose' in self.layer_dict:
                layer_time = self.layer_dict['ConvTranspose'].estimate(layer_info)
            elif layer_info['type'] == 'Pool' and 'Pool' in self.layer_dict:
                layer_time = self.layer_dict['Pool'].estimate(layer_info)
            elif layer_info['type'] == 'Add' and 'Add' in self.layer_dict:
                layer_time = self.layer_dict['Add'].estimate(layer_info)
            elif layer_info['type'] == 'FullyConnected' and 'FullyConnected' in self.layer_dict:
                layer_time = self.layer_dict['FullyConnected'].estimate(layer_info)
            elif layer_info['type'] == 'ConvPool' and 'ConvPool' in self.layer_dict:
                layer_time = self.layer_dict['ConvPool'].estimate(layer_info)
            elif layer_info['type'] == 'DataInput' and 'Input' in self.layer_dict:
                layer_time = self.layer_dict['Input'].estimate(layer_info)
            elif layer_info['type'] in self.layer_dict:
                layer_time = self.layer_dict[layer_info['type']].estimate(layer_info)
            else:
                layer_time = self.layer_dict['Base'].estimate(layer_info)
            result[layer_name] = layer_time
            if layer_time:
                sum_result = sum_result + layer_time
            try:
                gop = layer_info['num_ops'] 
            except:
                gop = 0
            try:
                n_i = layer_info['num_inputs'] 
            except:
                n_i = 0
            try:
                n_o = layer_info['num_outputs'] 
            except:
                n_o = 0
            try:
                n_w = layer_info['num_weights'] 
            except:
                n_w = 0
            result_pd.loc[len(result_pd)] = {"name":layer_name,"type":layer_info['type'],"time(ms)":layer_time,"num_ops":gop,"num_inputs":n_i,"num_outputs":n_o,"num_weights":n_w}

        return [sum_result, result, result_pd]

    def to_json(self,filename=None):
        """Store Hardware Estimator to json file"""
        layer_desc = {}
        layer_desc['name'] = self.name
        layer_desc['op_s'] = self.op_s
        layer_desc['bandwidth'] = self.bandwidth
        layer_desc['architecture'] = self.architecture
        for l in self.layer_dict.items():
            layer_desc[l[0]] = l[1].desc
            print(l)
            print(l[1].desc
            )
        if filename:
            with open(filename, 'w') as f:
                f.write(json.dumps(layer_desc, indent=4))

    @classmethod
    def from_json(cls,filename):
        """Reconstruct Estimator from json file"""

        with open(filename, 'r') as f:
            layer_desc = json.load(f)
        
        output_model = cls(layer_desc['name'], layer_desc['op_s'], layer_desc['bandwidth'], layer_desc['architecture'])

        print("Initial % d entries..." % len(layer_desc))
        del layer_desc['name']; del layer_desc['op_s']; del layer_desc['bandwidth']; del layer_desc['architecture']
        logging.debug(len(layer_desc))
        for l in layer_desc.items():
            #if l[0] in output_model.layer_classes:
            logging.debug(l[0])
            logging.debug(l[1])
            layer = l[1]
            if 'architecture' in layer:
                logging.debug("architecture defined")
                arc = layer['architecture']
            else:
                arc = None

            if not 'y_val' in layer:
                layer['y_val'] = None

            if layer['est_type'] == 'mixed':
                output_model.add_layer(layer['name'], layer['layer_type'], layer['est_type'], layer['op_s'], layer['bandwidth'],
                    architecture = layer['architecture'],
                    est_model = layer['est_model'],
                    conv_dict = layer['est_dict'],
                    y_val = layer['y_val'])
            elif layer['est_type'] == 'statistical':
                output_model.add_layer(layer['name'], layer['layer_type'], layer['est_type'], layer['op_s'], layer['bandwidth'],
                    est_model = layer['est_model'],
                    conv_dict = layer['est_dict'])
            elif layer['est_type'] == 'refined_roofline':
                output_model.add_layer(layer['name'], layer['layer_type'], layer['est_type'], layer['op_s'], layer['bandwidth'],
                    architecture = layer['architecture'])
            else:
                output_model.add_layer(layer['name'], layer['layer_type'], layer['est_type'], layer['op_s'], layer['bandwidth'],
                    architecture=arc)
        return output_model


def main():

    return True