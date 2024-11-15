from __future__ import print_function

import json
import logging
from functools import reduce
from pprint import pprint
import time

import numpy as np
import pandas as pd
from annette.estimation import layers

from onnx import helper

__author__ = "Matthias Wess"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"

logger = logging.getLogger(__name__)

class Layer_model():
    """Generates a Layer model that is used by the Estimation Tool.
    """

    onnx_transformations = {
        "Conv": {
            "out_type": "DepthwiseConv",
            "condition": {
                "value1": {
                    "name": "group"
                },
                "value2": {
                    "name": "output_shape",
                    "i": 3
                },
                "compare": "=="
            }
        }
    }

    def __init__(self, name, op_s, bandwidth, architecture=None):
        """Initialize the layer.

        Args:
            name ([type]): [description]
            op_s ([type]): [description]
            bandwidth ([type]): [description]
            architecture ([type], optional): [description]. Defaults to None.
        """
        self.name = name
        self.op_s = op_s
        self.bandwidth = bandwidth
        if architecture:
            self.architecture = architecture

        self.layer_classes = layers.__layer_classes__

        self.layer_dict = {}
        print(self.layer_dict)

    def add_layer(self, name, layer_type, est_type, op_s, bandwidth, architecture=None, est_model=None, conv_dict=None, diff_model=None, y_val='ops/s'):
        """add a layer to the model

        Args:
            name ([type]): [description]
            layer_type ([type]): [description]
            est_type ([type]): [description]
            op_s ([type]): [description]
            bandwidth ([type]): [description]
            architecture ([type], optional): [description]. Defaults to None.
            est_model ([type], optional): [description]. Defaults to None.
            conv_dict ([type], optional): [description]. Defaults to None.
            y_val (str, optional): [description]. Defaults to 'ops/s'.
        """
        # check if architecture available
        if est_type in ["mixed", "refined_roofline"]:
            if architecture is None:
                logging.error(
                    "could not build layer because no architecture defined")
                pass
        if est_type in ["mixed", "statistical"]:
            # check if conv_dict available
            if (conv_dict is None) or (est_model is None):
                logging.error(
                    "could not build layer because no conversion dictionary available")

        # check which type of layer
        if layer_type in self.layer_classes:
            # generate layer
            print("generate {}-layer".format(layer_type))
            # add to layer_dict
            self.layer_dict[layer_type] = self.layer_classes[layer_type](
                name, layer_type, est_type, op_s, bandwidth, architecture)

        else:
            # generate layer
            print("generate {}-layer".format(layer_type))
            # add to layer_dict
            self.layer_dict[layer_type] = self.layer_classes['Base'](
                name, layer_type, est_type, op_s, bandwidth, architecture)

        # check which type of estimation
        if est_type in ["mixed", "statistical"]:
            print("load statistical estimator: {}...".format(est_model))
            print("dictionary: {}...".format(conv_dict))
            self.layer_dict[layer_type].load_estimator(
                est_model, conv_dict, diff_model)

        self.layer_dict[layer_type].y_val = y_val

        pass

    def prepare_result_dataframe (self):
        logging.info('[layer_model.py | prepare_result_dataframe()]: Start.')
        result_pd = pd.DataFrame({"name": [], "type": [], "estimation_type": [], "time(ms)": []})
        result_pd['num_ops'] = np.nan
        result_pd['num_inputs'] = np.nan
        result_pd['num_outputs'] = np.nan
        result_pd['num_weights'] = np.nan
        result_pd['difficulty1'] = np.nan
        result_pd['difficulty2'] = np.nan
        result_pd['difficulty3'] = np.nan
        result_pd['difficulty4'] = np.nan
        return result_pd

    def estimate_model(self, model):
        """estimate the model

        Args:
            model (:obj:`annette.graph.AnnetteGraph`): annette network description

        Returns:
            (list) : [sum of time in ms, layer wise results [dict], layer-wise results [pandas.dataframe]] 
        """
        logging.debug(self.layer_dict)
        #sys.exit()
        result = {}
        sum_result = 0
        # print(model.model_spec['layers'])
        result_pd = self.prepare_result_dataframe()

        # Add info to layer stuff
        """Loop through Layers"""
        start = time.time()
        
        # for layer_name, layer_info in model.model_spec['layers'].items():
        for layer_name in model.topological_sort:
            layer_info = model.model_spec['layers'][layer_name]
            layer_info['time_ms'] = 0

            if layer_info['type'] in self.layer_dict:
                layer_info['time_ms'] = \
                    self.layer_dict[layer_info['type']].estimate(layer_info)
            else:
                layer_info['time_ms'] = self.layer_dict['Base'].estimate(layer_info)

            def try_read(argument):
                try:
                    result = layer_info[argument]
                except:
                    result = layer_info['time_ms']
                return result 

            result[layer_name] = layer_info['time_ms'] 
            sum_result = sum_result + layer_info['time_ms'] 

            gop = try_read('num_ops')
            n_i = try_read('num_inputs')
            n_o = try_read('num_outputs')
            n_w = try_read('num_weights')
            n_w = try_read('num_weights')
            diff = try_read('difficulty1')
            diff2 = try_read('difficulty2')
            diff3 = try_read('difficulty3')
            diff4 = try_read('difficulty4')
            result_pd.loc[len(result_pd)] = {"name": layer_name, "type": layer_info['type'],
                                             "time(ms)": layer_info['time_ms'], "num_ops": gop,
                                             "num_inputs": n_i, "num_outputs": n_o, "num_weights": n_w,
                                             "difficulty1": diff, "difficulty2": diff2, "difficulty3": diff3, "difficulty4": diff4}

        end = time.time()
        print("Layermodel executed in ", end-start)

        return [sum_result, result, result_pd]

    def estimate_model_onnx (self, onnx_model):
        logger.debug('[estimate_model_onnx]: Start. onnx_model = %s' % str(onnx_model.network_name))
        onnx_graph = onnx_model.get_node_graph()
        result = {}
        sum_result = 0
        result_pd = self.prepare_result_dataframe()

        time_start = time.time()

        def try_read_node_attribute (node, argument):
            logger.debug('[estimate_model_onnx]: Try to read the attribute. node.name = %s, argument = %s' % (str(node.name), str(argument)))
            try:
                result = helper.get_node_attr_value(node, argument)
            except:
                logger.info('[estimate_model_onnx]: Attribute could not be read. Falling back to the time_ms attribute. node.name = %s, argument = %s' % (str(node.name), str(argument)))
                result = helper.get_node_attr_value(node, 'time_ms')
            return result
        
        def perform_value_compare (compare, value1, value2):
            logger.debug('[perform_value_compare]: Start. condition = %s, value1 = %s, value2 = %s' % (str(compare), str(value1), str(value2)))
            # The following lines check, based on the condition (e.g., == or >), if the two values fulfill this condition.
            if compare == "==" and str(value1) == str(value2):
                return True
            elif compare == ">" and float(value1) > float(value2):
                return True
            elif compare == "<" and float(value1) < float(value2):
                return True
            elif compare == ">=" and float(value1) >= float(value2):
                return True
            elif compare == "<=" and float(value1) <= float(value2):
                return True
            else:
                return False
            
        def get_attribute_value_of_node (node_nums, condition):
            attribute_value = node_nums[condition["name"]]
            if 'i' in condition:
                if int(condition['i']) <= (len(attribute_value) - 1):
                    logger.debug('[get_attribute_value_of_node]: Getting the specified index of the attribute. condition[\'i\'] = %s, attribute_value = %s' % (str(condition['i']), str(attribute_value)))
                    return attribute_value[int(condition['i'])]
                else:
                    logger.error('[get_attribute_value_of_node]: Provided index is out of bounds. Setting the attribute_value to 0. condition[\'i\'] = %s, attribute_value = %s' % (str(condition['i']), str(attribute_value)))
                    return 0
            else:
                return attribute_value

        def check_transformation_condition (node, transformation):
            node_nums = onnx_model.compute_nums_for_node(node)
            condition = transformation['condition']
            if not (condition["value1"]["name"] in node_nums):
                logger.warning('[check_transformation_condition]: Attribute does not exist on this node. condition["value1"]["name"] = %s, node_nums = %s' % (str(condition["value1"]["name"]), str(node_nums)))
                return False
            if not (condition["value2"]["name"] in node_nums):
                logger.warning('[check_transformation_condition]: Attribute does not exist on this node. condition["value2"]["name"] = %s, node_nums = %s' % (str(condition["value2"]["name"]), str(node_nums)))
                return False
            value1 = get_attribute_value_of_node(node_nums, condition["value1"])
            value2 = get_attribute_value_of_node(node_nums, condition["value2"])
            compare_result = perform_value_compare(condition["compare"], value1, value2)
            logger.debug('[check_transformation_condition]: Result has been evaluated. compare_result = %s' % str(compare_result))
            return compare_result

        for node in onnx_graph:
            logger.debug('[estimate_model_onnx]: Processing node. node.name = %s' % str(node.name))
            node_estimation = 0
            input_shape = onnx_model.get_node_input_shape(node)
            output_shape = onnx_model.get_node_output_shape(node)
            attributes = onnx_model.get_node_attributes(node)
            num_weights = onnx_model.get_number_of_node_weights(node)
            num_operations = onnx_model.get_node_flops_by_name(node.name)

            node_type = node.op_type
            if (node_type in self.onnx_transformations) and (check_transformation_condition(node, self.onnx_transformations[node.op_type])):
                node_type = self.onnx_transformations[node.op_type]['out_type']

            if node_type in self.layer_dict:
                logger.debug('[estimate_model_onnx]: Starting the estimation based on the %s class. node.name = %s' % (str(node_type), str(node.name)))
                node_estimation = self.layer_dict[node_type].estimate_onnx(node, input_shape, output_shape, attributes, num_weights, num_operations)
            else:
                logger.debug('[estimate_model_onnx]: Starting the estimation based on the Base class. node.name = %s' % str(node.name))
                node_estimation = self.layer_dict['Base'].estimate_onnx(node, input_shape, output_shape, attributes, num_weights, num_operations)

            result[node.name] = node_estimation
            sum_result = sum_result + node_estimation

            node.attribute.append(helper.make_attribute('time_ms', node_estimation))

            logger.debug('[estimate_model_onnx]: Finalized the estimation and saved the estimation to the node. node_estimation = %s, sum_result = %s' % (str(node_estimation), str(sum_result)))

            gop = try_read_node_attribute(node, 'num_ops')
            n_i = try_read_node_attribute(node, 'num_inputs')
            n_o = try_read_node_attribute(node, 'num_outputs')
            n_w = try_read_node_attribute(node, 'num_weights')
            diff = try_read_node_attribute(node, 'difficulty1')
            diff2 = try_read_node_attribute(node, 'difficulty2')
            diff3 = try_read_node_attribute(node, 'difficulty3')
            diff4 = try_read_node_attribute(node, 'difficulty4')

            result_pd.loc[len(result_pd)] = {
                "name": node.name,
                "type": node.op_type,
                "estimation_type": node_type,
                "time(ms)": node_estimation,
                "num_ops": gop,
                "num_inputs": n_i,
                "num_outputs": n_o,
                "num_weights": n_w,
                "difficulty1": diff,
                "difficulty2": diff2,
                "difficulty3": diff3,
                "difficulty4": diff4
            }
        time_end = time.time()
        print('Layermodel executed in ', time_end - time_start)

        return [sum_result, result, result_pd]

    def to_json(self, filename=None):
        """Store Hardware Estimator to json file

        Args:
            filename ([type], optional): [description]. Defaults to None.
        """
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

    @ classmethod
    def from_json(cls, filename):
        """Reconstruct Estimator from json file
        """

        with open(filename, 'r') as f:
            layer_desc = json.load(f)

        output_model = cls(layer_desc['name'], layer_desc['op_s'],
                           layer_desc['bandwidth'], layer_desc['architecture'])

        print("Initial {} entries...".format(len(layer_desc)))
        del layer_desc['name']
        del layer_desc['op_s']
        del layer_desc['bandwidth']
        del layer_desc['architecture']
        if 'hardware_description' in layer_desc:
            del layer_desc['hardware_description']
        logging.debug(len(layer_desc))
        for l in layer_desc.items():
            # if l[0] in output_model.layer_classes:
            logging.debug(l[0])
            logging.debug(l[1])
            layer = l[1]
            if 'architecture' in layer:
                logging.debug("architecture defined")
                arc = layer['architecture']
            else:
                arc = None

            if 'y_val' not in layer.keys():
                layer['y_val'] = None
            if 'difficulty' not in layer.keys():
                print(layer)
                layer['difficulty'] = None
            else:
                print(layer)

            if layer['est_type'] == 'mixed':
                output_model.add_layer(layer['name'], layer['layer_type'], layer['est_type'], layer['op_s'], layer['bandwidth'],
                                       architecture=layer['architecture'],
                                       est_model=layer['est_model'],
                                       conv_dict=layer['est_dict'],
                                       y_val=layer['y_val'],
                                       diff_model=layer['difficulty'])
            elif layer['est_type'] == 'statistical':
                output_model.add_layer(layer['name'], layer['layer_type'], layer['est_type'], layer['op_s'], layer['bandwidth'],
                                       est_model=layer['est_model'],
                                       conv_dict=layer['est_dict'],
                                       diff_model=layer['difficulty'],
                                       y_val=layer['y_val'])
            elif layer['est_type'] == 'refined_roofline':
                output_model.add_layer(layer['name'], layer['layer_type'], layer['est_type'], layer['op_s'], layer['bandwidth'],
                                       architecture=layer['architecture'])
            else:
                output_model.add_layer(layer['name'], layer['layer_type'], layer['est_type'], layer['op_s'], layer['bandwidth'],
                                       architecture=arc)
        return output_model

def main():
    """main function that runs the main loop
    """

    return True
