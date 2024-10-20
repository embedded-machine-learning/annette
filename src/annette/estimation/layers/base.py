from __future__ import print_function

import logging
import pickle
from functools import reduce
from pprint import pprint

import numpy as np
from onnx import helper
from annette import get_database

logger = logging.getLogger(__name__)

class BaseLayer(object):
    """BaseLayer estimation"""

    def __init__(self, name, layer_type="Base", est_type="roofline", op_s=1e9, bandwidth=1e9, architecture=None, y_val='ops/s'):
        self.name = name
        self.layer_type = layer_type
        self.estimation = est_type
        self.y_val = y_val
        # Model parameters
        self.op_s = op_s 
        self.bandwidth = bandwidth 
        if not architecture:
            self.architecture = {"bit_act": 8, "bit_weights": 8}
        else:
            self.architecture = architecture
        self.confidence = 0.8
        # Layer parameters
        self.layer = {}
        self.layer['num_inputs'] = None
        self.layer['num_outputs'] = None
        self.layer['num_ops'] = None
        self.layer['parents'] = None
        self.layer['y_val'] = y_val
        
        # Layer description dictionary, add information for rebuilding here
        self.desc = self.gen_dict()

    @staticmethod
    def compute_nums(layer):
        """Compute Num Parameters for Base Layer prediction"""
        try:
            if not type(layer['input_shape'][0]) == list:
                # In case the node has only one input, we want to use this calculation method
                # This case will apply to most nodes, hence why it's first in the if-statement.
                layer['num_inputs'] = reduce(lambda x, y: x*y, layer['input_shape'][1:])
            else:
                # In case the node has multiple inputs, we want to use this calculation method
                # Case: Multiple input tensors of same size (e.g. Concat layer)
                layer['num_inputs'] = 2 * reduce(lambda x, y: x*y, layer['input_shape'][0][1:])
        except:
            layer['num_inputs'] = 0
        try:
            layer['num_outputs'] = reduce(lambda x, y: x*y, layer['output_shape'][1:])
        except:
            layer['num_outputs'] = 0
        # It's possible with the ONNX estimation, that num_ops and num_weights are not set initially. In this case, initialize them as 0.
        if not ('num_ops' in layer.keys()):
            layer['num_ops'] = 0
        if not ('num_weights' in layer.keys()):
            layer['num_weights'] = 0

        return layer

    def compute_efficiency(self, unrolled, eff, div, mod, par = None, alpha = None, replication = False):
        """Compute layer efficiency for one unrolled parameter.

        Args:
            unrolled (int): unrolled parameter e.g. self.layer['output_shape'][1] for height
            eff (str): key of efficiency to write to e.g. 'h_eff'
            div (str): name of divider result. Defaults to None.
            mod (str): name of mod result. Defaults to None.
            par (str, optional): name of efficiency parameter in architecture description. Defaults to None.
            alpha (str, optional): name of the alpha parameter in the architecture description. Defaults to None.
            replication (bool, optional): replication for unrolled < par enabled. Defaults to True.
        """

        if self.architecture:
            logging.debug(self.architecture)
            if par in self.architecture:
                self.layer[eff] = 1
                self.layer[mod] = 0
                self.layer[div] = unrolled
                if self.architecture[par] >= 1:
                    self.layer[mod] = unrolled % self.architecture[par]
                    self.layer[div] = unrolled // self.architecture[par]

                    if unrolled < self.architecture[par] and replication == True:
                        self.layer[eff] = 1
                    else:  # equals formula 3 in paper:
                        self.layer[eff] = unrolled / (np.ceil(unrolled/self.architecture[par]) * self.architecture[par])
            
                logging.debug(f'{eff}: {self.layer[eff]}')
                if alpha in self.architecture:
                    self.layer[eff] = 1 / ((1-self.architecture[alpha]) + 1/self.layer[eff] * (self.architecture[alpha]))
                logging.debug(f'{eff} with alpha: {self.layer[eff]}')
            else:
                self.layer[eff] = 1
                logging.error(f'{par} not in architecture!')
        else:
            self.layer[eff] = 1
            logging.error('No architecture available!')

    def compute_parameters(self, layer=None):
        """Compute Parameters for Base Layer prediction"""
        self.layer = self.compute_nums(self.layer)
        return self.layer

    def estimate(self, layer=None):
        """Returns estimated layer execution time (ms)"""
        logging.info(f'Estimation Type: {self.estimation}')

        if hasattr(self, "estimate_" + self.estimation):
            self.layer = layer
            self.compute_parameters()
            logging.debug(f'Current layer: {self.layer}')
            func = getattr(self, "estimate_" + self.estimation)
            r = func()
            return r
        else:
            logging.error(f'No estimator for {self.estimation} implemented.')
            r = self.estimate_roofline()
            return r

    def save_parameters_to_node (self, node):
        logger.debug('[estimate_onnx]: Start. node.name = %s' % str(node.name))
        node.attribute.append(helper.make_attribute('num_ops', self.layer['num_ops']))
        node.attribute.append(helper.make_attribute('num_inputs', self.layer['num_inputs']))
        node.attribute.append(helper.make_attribute('num_outputs', self.layer['num_outputs']))
        node.attribute.append(helper.make_attribute('num_weights', self.layer['num_weights']))
        if ('difficulty' in self.layer):
            node.attribute.append(helper.make_attribute('difficulty', self.layer['difficulty']))
        if ('difficulty2' in self.layer):
            node.attribute.append(helper.make_attribute('difficulty2', self.layer['difficulty2']))
        if ('difficulty3' in self.layer):
            node.attribute.append(helper.make_attribute('difficulty3', self.layer['difficulty3']))
        if ('difficulty4' in self.layer):
            node.attribute.append(helper.make_attribute('difficulty4', self.layer['difficulty4']))

    def estimate_onnx (self, node, input_shape, output_shape, attributes, num_weights, num_operations):
        logger.debug('[estimate_onnx]: Start. node.name = %s, input_shape = %s, output_shape = %s, num_weights = %s, num_operations = %s' % (str(node.name), str(input_shape), str(output_shape), str(num_weights), str(num_operations)))
        self.node = node
        self.layer['input_shape'] = input_shape
        self.layer['output_shape'] = output_shape
        self.layer['num_weights'] = num_weights
        self.layer['num_ops'] = num_operations
        for attribute in attributes:
            logger.debug('[estimate_onnx]: Processing attribute. attribute = %s, attributes[attribute] = %s' % (str(attribute), str(attributes[attribute])))
            self.layer[attribute] = attributes[attribute]
        self.compute_parameters()
        if hasattr(self, "estimate_" + self.estimation):
            func = getattr(self, "estimate_" + self.estimation)
            logger.debug('[estimate_onnx]: Starting the estimation of the node. node.name = %s, func = %s' % (str(node.name), str(func.__name__)))
            r = func()
        else:
            logger.debug('[estimate_onnx]: Starting the estimation of the node. node.name = %s, self.estimation = %s' % (str(node.name), str(self.estimation)))
            r = self.estimate_roofline()
        self.save_parameters_to_node(node)
        logger.debug('[estimate_onnx]: Estimation successful! Saved parameters to the node. r = %s, node = %s' % (str(r), str(node)))
        return r

    def estimate_roofline(self):
        """Returns roofline estimated BaseLayer execution time (ms)"""
        self.layer['data_bytes'] = ((
            (self.layer['num_outputs'] + self.layer['num_inputs']) 
            * self.architecture['bit_act']
            + self.layer['num_weights'] * self.architecture['bit_weights'])
            / 8)
        self.layer['data_roof'] = self.layer['data_bytes'] / self.bandwidth
        self.layer['op_roof'] = self.layer['num_ops'] / self.op_s
        self.layer['time_ms'] = np.max([self.layer['op_roof'], self.layer['data_roof']]) * 1000  # to milliseconds

        logging.debug(self.layer['op_roof'])
        logging.debug(self.layer['data_roof'])
        logging.debug(self.layer['time_ms'])

        if self.layer['op_roof'] > self.layer['data_roof']:
            logging.debug("Op roof")
        else:
            logging.debug("Data roof")

        return self.layer['time_ms']

    def estimate_refined_roofline(self):
        """Returns roofline estimated BaseLayer execution time (ms)"""
        self.estimate_roofline()
        return self.refine_value()

    def estimate_statistical(self):
        self.estimate_roofline()
        result = [1.]
        if hasattr(self, 'est_dict') and hasattr(self, 'est_model'):
            vector = self.build_vector(self.est_dict)
            result = self.est_model.predict(vector)
            if self.y_val == 's/ops':
                self.layer['time_ms'] = self.layer['num_ops'] * result[0] / 1e6
            else:
                self.layer['time_ms'] = self.layer['num_ops'] / result[0] * 1e3
        else:
            logging.error('Layer type does not have est_dict or est_model!')
        y_hat_test = result
        if hasattr(self, 'est_dict') and hasattr(self, 'diff_model'):
            if self.diff_model is None:
                logging.error('No difficulty model defined!')
                return self.layer['time_ms']
            else:
                vector = self.build_vector(self.est_dict)
                sigmas = self.diff_model['diff'].apply(vector)
                sigmas2 = self.diff_model['diff2'].apply(vector)
                r = np.arange(0.95, 0.04, -0.05)
                self.layer['difficulty1'] = [0]*(len(r)*2+1)
                self.layer['difficulty3'] = [0]*(len(r)*2+1)
                for i, a in enumerate(r):

                    if self.y_val == 's/ops':
                        ints = self.diff_model['reg'].predict(y_hat_test,
                                                        sigmas=sigmas,
                                                        confidence=a)[0]
                        ints2 = self.diff_model['reg2'].predict(y_hat_test*self.layer['num_ops'],
                                                        sigmas=sigmas2,
                                                        confidence=a)[0]
                        self.layer['difficulty1'][i] = self.layer['num_ops'] * ints[0] / 1e6
                        self.layer['difficulty1'][len(r)*2-i] = self.layer['num_ops'] * ints[1] / 1e6
                        self.layer['difficulty3'][i] = ints2[0] / 1e6
                        self.layer['difficulty3'][len(r)*2-i] = ints2[1] / 1e6
                    else:
                        ints = self.diff_model['reg'].predict(y_hat_test,
                                                        sigmas=sigmas,
                                                        confidence=a)[0]
                        ints2 = self.diff_model['reg2'].predict(y_hat_test*self.layer['num_ops'],
                                                        sigmas=sigmas2,
                                                        confidence=a)[0]
                        self.layer['difficulty1'][i] = self.layer['num_ops'] / ints[1] * 1e3
                        self.layer['difficulty1'][len(r)*2-i] = self.layer['num_ops'] / ints[0] * 1e3
                        self.layer['difficulty3'][i] = 1 / ints2[1] * 1e3
                        self.layer['difficulty3'][len(r)*2-i] = 1 / ints2[0] * 1e3
                self.layer['difficulty1'][len(r)] = self.layer['time_ms']
                self.layer['difficulty3'][len(r)] = self.layer['time_ms']
                tmp = 0
                for i in range(len(self.layer['difficulty1'])-1,-1,-1):
                    # if self.layer['difficulty'][i] is inf
                    if self.layer['difficulty1'][i] == np.inf:
                        self.layer['difficulty1'][i] = tmp
                    tmp = self.layer['difficulty1'][i]

            if self.diff_model2 is None:
                logging.error('No difficulty_std model defined!')
                return self.layer['time_ms']
            else:
                vector = self.build_vector(self.est_dict)
                sigmas = self.diff_model2['diff'].apply(vector)
                sigmas2 = self.diff_model2['diff2'].apply(vector)
                r = np.arange(0.95, 0.04, -0.05)
                self.layer['difficulty2'] = [0]*(len(r)*2+1)
                self.layer['difficulty4'] = [0]*(len(r)*2+1)
                for i, a in enumerate(r):
                    if self.y_val == 's/ops':
                        ints = self.diff_model2['reg'].predict(y_hat_test,
                                                        sigmas=sigmas,
                                                        confidence=a)[0]
                        ints2 = self.diff_model2['reg2'].predict(y_hat_test*self.layer['num_ops'],
                                                        sigmas=sigmas2,
                                                        confidence=a)[0]
                        self.layer['difficulty2'][i] = self.layer['num_ops'] * ints[0] / 1e6
                        self.layer['difficulty2'][len(r)*2-i] = self.layer['num_ops'] * ints[1] / 1e6
                        self.layer['difficulty4'][i] = ints2[0] / 1e6
                        self.layer['difficulty4'][len(r)*2-i] = ints2[1] / 1e6
                    else:
                        ints = self.diff_model2['reg'].predict(y_hat_test,
                                                        sigmas=sigmas,
                                                        confidence=a)[0]
                        ints2 = self.diff_model2['reg2'].predict(y_hat_test*self.layer['num_ops'],
                                                        sigmas=sigmas2,
                                                        confidence=a)[0]
                        self.layer['difficulty2'][i] = self.layer['num_ops'] / ints[1] * 1e3
                        self.layer['difficulty2'][len(r)*2-i] = self.layer['num_ops'] / ints[0] * 1e3
                        self.layer['difficulty4'][i] = 1 / ints2[1] * 1e3
                        self.layer['difficulty4'][len(r)*2-i] = 1 / ints2[0] * 1e3
                self.layer['difficulty2'][len(r)] = self.layer['time_ms']
                self.layer['difficulty4'][len(r)] = self.layer['time_ms']
                tmp = 0
                for i in range(len(self.layer['difficulty2'])-1,-1,-1):
                    # if self.layer['difficulty'][i] is inf
                    if self.layer['difficulty2'][i] == np.inf:
                        self.layer['difficulty2'][i] = tmp
                    tmp = self.layer['difficulty2'][i]
            
        return self.layer['time_ms']

    def estimate_mixed(self):
        self.estimate_statistical()
        return self.refine_value()

    def refine_value(self):
        if hasattr(self, 'compute_eff'):
            func = getattr(self, 'compute_eff')
            func()

        if 'eff' in self.layer.keys():
            if self.layer['op_roof'] > self.layer['data_roof']:
                logging.debug(f"Refining with efficiency: {self.layer['eff']}")
                self.layer['time_ms'] /= (self.layer['eff'])
        else:
            logging.error('No efficiency defined for this layer type!')

        return self.layer['time_ms']

    def load_estimator(self, est_model=None, est_dict=None, diff_model=None):
        if est_model != None:
            self.est_model = pickle.load(open(get_database(est_model), 'rb'))
            self.desc['est_model'] = est_model
            try:
                est_model2 = est_model.replace('.sav', '2.sav')
                self.est_model2 = pickle.load(open(get_database(est_model2), 'rb'))
                self.desc['est_model2'] = est_model2
            except:
                self.est_model2 = None
                self.desc['est_model2'] = None
        else:
            return False

        if diff_model is not None:
            self.diff_model = pickle.load(open(get_database(diff_model), 'rb'))
            self.desc['difficulty'] = diff_model
            try:
                diff_model2 = diff_model.replace('.sav', '_std.sav')
                self.diff_model2 = pickle.load(open(get_database(diff_model2), 'rb'))
                self.desc['difficulty_std'] = diff_model2
            except:
                self.diff_model2 = None
                self.desc['difficulty_std'] = None
        else:
            self.diff_model = None

        self.est_dict = est_dict
        self.desc['est_dict'] = est_dict

        return True

    def build_vector(self, in_vector, degree=None):
        """Build Estimation Vector"""
        vector = np.zeros([1,len(in_vector)])
        for k, v in in_vector.items():
            k_int = int(k)
            if isinstance(v, dict):
                if (v['name'] in self.layer) and (self.layer[v['name']] != None):
                    vector[0, k_int] = self.layer[v['name']][v['i']]
                if 'dec' in v.keys():
                    vector[0, k_int] = vector[0, k_int] - v['dec'] 
            elif isinstance(v, str):
                vector[0, k_int] = self.layer[v]
            else:
                vector[0, k_int] = v

        return vector

    def gen_dict(self, filename=None):
        desc = {"name": self.name,
                "layer_type": self.layer_type,
                "est_type": self.estimation,
                "op_s": self.op_s,
                "bandwidth": self.bandwidth,
                "architecture": self.architecture,
                "y_val": self.y_val}
        return desc
