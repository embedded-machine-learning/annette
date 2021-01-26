from __future__ import print_function
from pprint import pprint
from functools import reduce
import pickle
import numpy as np
import pandas as pde
from annette.estimation.layers.base import BaseLayer

class DepthwiseConvLayer(BaseLayer):
    """DepthwiseConvLayer estimation"""

    def __init__(self, name, layer_type = "Conv", est_type = "roofline", op_s = 1*1e9, bandwidth = 1*1e9, architecture = None):
        self.name = name
        self.layer_type = layer_type
        self.estimation = est_type

        # Model parameters
        self.op_s = op_s 
        self.bandwidth = bandwidth 
        if architecture:
            self.architecture = architecture 
        else:
            self.architecture = {}
        if not "bit_act" in self.architecture:
            self.architecture["bit_act"] = 8
        if not "bit_weights" in self.architecture:
            self.architecture["bit_weights"] = 8
        self.y_val = 'ops/s'

        # Layer stuff
        self.num_weights = None
        self.num_inputs = None
        self.num_outputs = None
        self.num_ops= None
        self.parents = None

        # Layer description dictionary, add information for rebuilding here
        self.desc = self.gen_dict()

    def estimate(self, layer = None):
        """return estimated ConvLayer execution time (ms)"""
        print("Estimation Type: " + self.estimation)
        if hasattr(self, "estimate_" + self.estimation):
            self.layer = layer
            func = getattr(self, "estimate_" + self.estimation)
            r = func()
            return r
        else:
            print("No " + self.estimation + " Estimator implemented")
            return 0

    @staticmethod
    def compute_nums(layer):
        """Compute Num Parameters for Convolution Layer prediction"""

        layer['num_weights'] = reduce(lambda x, y: x*y, layer['kernel_shape'])
        layer['num_outputs'] = reduce(lambda x, y: x*y, layer['output_shape'][1:])
        layer['num_inputs'] = reduce(lambda x, y: x*y, layer['output_shape'][1:3])*layer['kernel_shape'][2]*reduce(lambda x, y: x*y, layer['strides'][1:])
        print(reduce(lambda x, y: x*y, layer['output_shape'][1:3]))
        print(layer['kernel_shape'][2])
        print(reduce(lambda x, y: x*y, layer['strides'][1:]))
        print(layer['input_shape'])

        layer['num_ops'] = (
            layer['num_weights'] * layer['output_shape'][1] * layer['output_shape'][2]
            )*2
        
        return layer

    def compute_parameters(self, layer = None):
        """Compute Parameters for Convolution Layer prediction"""

        self.layer['num_weights'] = reduce(lambda x, y: x*y, self.layer['kernel_shape'])
        self.layer['num_outputs'] = reduce(lambda x, y: x*y, self.layer['output_shape'][1:])
        self.layer['num_inputs'] = reduce(lambda x, y: x*y, self.layer['output_shape'][1:3])*self.layer['kernel_shape'][2]*reduce(lambda x, y: x*y, self.layer['strides'][1:])
        print(reduce(lambda x, y: x*y, self.layer['output_shape'][1:3]))
        print(self.layer['kernel_shape'][2])
        print(reduce(lambda x, y: x*y, self.layer['strides'][1:]))
        print(self.layer['input_shape'])

        self.layer['num_ops'] = (
            self.layer['num_weights'] * self.layer['output_shape'][1] * self.layer['output_shape'][2]
            )*2

        try:
            if self.architecture:
                if 'h_par' in self.architecture:
                    if self.architecture['h_par'] < 1:
                        self.layer['h_mod'] = 1
                        self.layer['h_div'] = self.layer['output_shape'][1] - 1
                        self.layer['h_eff'] = 1
                    else:
                        self.layer['h_mod'] = (self.layer['output_shape'][1]-1)%self.architecture['h_par']+1
                        self.layer['h_div'] = (self.layer['output_shape'][1]-self.layer['h_mod'])/self.architecture['h_par']
                        if self.layer['output_shape'][1] < self.architecture['h_par']:
                            self.layer['h_eff'] = 1
                        else:
                            self.layer['h_eff'] = self.layer['output_shape'][1]/(np.ceil(self.layer['output_shape'][1]/self.architecture['h_par'])*self.architecture['h_par'])
                
                    print("h_eff",self.layer['h_eff'])
                    if 'h_alpha' in self.architecture:
                        self.layer['h_eff'] = 1/(1-self.architecture['h_alpha'] + 1/self.layer['h_eff'] * (self.architecture['h_alpha']))
                    print("h_eff",self.layer['h_eff'])

                    
                if 'c_par' in self.architecture:
                    if self.architecture['c_par'] < 1:
                        self.layer['c_mod'] = 0
                        self.layer['c_div'] = self.layer['kernel_shape'][2]
                        self.layer['c_eff'] = 1
                    else:
                        self.layer['c_mod'] = (self.layer['kernel_shape'][2]-1)%self.architecture['c_par']+1
                        self.layer['c_div'] = (self.layer['kernel_shape'][2]-self.layer['c_mod'])/self.architecture['c_par']
                        if self.layer['kernel_shape'][2] < self.architecture['c_par']:
                            self.layer['c_eff'] = 1
                        else:
                            self.layer['c_eff'] = self.layer['kernel_shape'][2]/(np.ceil(self.layer['kernel_shape'][2]/self.architecture['c_par'])*self.architecture['c_par'])
                    print("c_eff",self.layer['c_eff'])
                    if 'c_alpha' in self.architecture:
                        self.layer['c_eff'] = 1/(1-self.architecture['c_alpha'] + 1/self.layer['c_eff'] * (self.architecture['c_alpha']))
                    print("c_eff",self.layer['c_eff'])

                if 'f_par' in self.architecture:
                    if self.architecture['f_par'] < 1:
                        self.layer['f_mod'] = 1
                        self.layer['f_div'] = self.layer['kernel_shape'][3] - 1 
                        self.layer['f_eff'] = 1
                    else:
                        self.layer['f_mod'] = (self.layer['kernel_shape'][3]-1)%self.architecture['f_par']+1
                        self.layer['f_div'] = (self.layer['kernel_shape'][3]-self.layer['f_mod'])/self.architecture['f_par']
                        if self.layer['kernel_shape'][3] < self.architecture['f_par']:
                            self.layer['f_eff'] = 1
                        else:
                            self.layer['f_eff'] = self.layer['kernel_shape'][3]/(np.ceil(self.layer['kernel_shape'][3]/self.architecture['f_par'])*self.architecture['f_par'])
                    print("f_eff",self.layer['f_eff'])
                    if 'f_alpha' in self.architecture:
                        self.layer['f_eff'] = 1/(1-self.architecture['f_alpha'] + 1/self.layer['f_eff'] * (self.architecture['f_alpha']))
                    print("f_eff",self.layer['f_eff'])
            else:
                print("noarch")
        except:
            print("noarch")


        #print(self.layer)
        return self.layer

    def estimate_roofline(self):
        """returns roofline estimated ConvLayer execution time (ms)"""
        print("roofline estimation")
        self.layer = self.compute_parameters(self.layer)
        op_roof = self.layer['num_ops'] / self.op_s
        print(self.op_s)
        data_bytes = (
            (self.layer['num_inputs'] + self.layer['num_outputs'] )* self.architecture['bit_act']
            + self.layer['num_weights'] * self.architecture['bit_weights']) / 8
        print("Architecture: ", self.architecture)
        data_roof = data_bytes / self.bandwidth
        time_ms = np.max([op_roof, data_roof])*1000 # to milliseconds
        if op_roof > data_roof:
            print("OP Roof")
        else:
            print("Data Roof")
        print(op_roof)
        print(data_roof)
        print(time_ms)
        return time_ms

    def estimate_refined_roofline(self):
        """returns roofline estimated ConvLayer execution time (ms)"""
        print("refined roofline estimation")
        self.layer = self.compute_parameters(self.layer)
        op_roof = self.layer['num_ops'] / self.op_s
        data_bytes = (
            (self.layer['num_inputs'] + self.layer['num_outputs'] )* self.architecture['bit_act']
            + self.layer['num_weights'] * self.architecture['bit_weights']) / 8
        print("Architecture: ", self.architecture)
        data_roof = data_bytes / self.bandwidth
        time_ms = np.max([op_roof, data_roof])*1000 # to milliseconds
        if op_roof > data_roof:
            print("OP Roof")
            time_ms = time_ms / (self.layer['h_eff']*self.layer['c_eff']*self.layer['f_eff'])
        else:
            print("Data Roof")
        print(op_roof)
        print(data_roof)
        print(time_ms)
        return time_ms

    def estimate_statistical(self):
        print("statistical estimation")
        self.layer = self.compute_parameters(self.layer)
        vector = np.zeros([1,len(self.est_dict)])
        print(self.layer)
        for i in self.est_dict.items():
            if isinstance(i[1], dict):
                vector[0,int(i[0])] = self.layer[i[1]['name']][i[1]['i']] 
                if 'dec' in i[1].keys():
                    vector[0,i[0]] = vector[0,i[0]] - i[1]['dec'] 
            elif isinstance(i[1], str):
                vector[0,int(i[0])] = self.layer[i[1]]
            else:
                vector[0,int(i[0])] = i[1]
        print(vector)
        result = self.est_model.predict(vector)
        print(result)
        time_ms = self.layer['num_ops']/result[0]*1e3

        op_roof = self.layer['num_ops'] / self.op_s
        data_roof = (self.layer['num_inputs'] + self.layer['num_outputs']) / self.bandwidth
        #time_ms = np.max([op_roof, data_roof])*1000 # to milliseconds
        if op_roof > data_roof:
            print("OP Roof")
            #time_ms = time_ms / (self.layer['h_eff']*self.layer['c_eff']*self.layer['f_eff'])
        else:
            print("Data Roof")

        print(time_ms)
        return time_ms

    def estimate_mixed(self):
        print("mixed estimation")
        self.layer = self.compute_parameters(self.layer)
        #print(self.layer)
        #print(self.layer['num_weights'])
        #print(self.est_model)
        #print(self.est_dict)
        #ops,num_inputs,num_outputs,input_shape[1], input_shape[1]-1, 1, input_shape[2], input_shape[3], input_shape[3]-1, 1, output_shape[3], output_shape[3]-1, 1, kernel_shape[1], kernel_shape[2], num_weights]).reshape(1,-1)}
        vector = self.build_vector(self.est_dict)
        print(vector)
        print(vector.shape)
        result = self.est_model.predict(vector)
        print(result)
        print(self.y_val)

        op_roof = self.layer['num_ops'] / self.op_s
        data_roof = (self.layer['num_inputs'] + self.layer['num_outputs']) / self.bandwidth
        time_ms = np.max([op_roof, data_roof])*1000 # to milliseconds
        if self.y_val == 'time(ms)':
            time_ms = result[0]
        else:
            time_ms = self.layer['num_ops']/result[0]*1e3
        if op_roof > data_roof:
            print("OP Roof")
            time_ms = time_ms 
        else:
            print("Data Roof")

        #time_ms_stat = self.layer['num_ops']/result[0]*1e3
        #time_ms = np.max([time_ms, time_ms_stat])

        return time_ms

    def load_estimator(self, est_model=None, est_dict=None):
        if est_model != None:
            self.est_model = pickle.load(open(est_model, 'rb'))
            self.desc['est_model'] = est_model
        else:
            self.est_model = pickle.load(open('database/conv2d_all.sav', 'rb'))

        self.est_dict = est_dict
        self.desc['est_dict'] = est_dict
        print(self.est_dict)
    