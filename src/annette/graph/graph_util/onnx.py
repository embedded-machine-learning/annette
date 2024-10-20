import os, logging

import numpy as np

from onnx import load, GraphProto, NodeProto, ValueInfoProto
from onnx.defs import get_schema
from onnx.helper import get_attribute_value, get_node_attr_value
from onnx.numpy_helper import to_array
from onnxsim import simplify
from onnx_tool import Model as onnx_tool_model

from numpy import ndarray
from pathlib import PosixPath
from google.protobuf.internal.containers import RepeatedCompositeFieldContainer, RepeatedScalarFieldContainer
from onnx_tool import Node as onnx_tool_node
from annette.estimation.mapping_model import Optimizer

logger = logging.getLogger(__name__)

class ONNX ():
    # This Boolean determines whether the ONNX model provided should first be simplified or not. onnxsim is used for the simplification.
    use_simplify: bool = True
    # This Boolean determines whether onnx_tool should be used for the calculation of the MACs and FLOPs.
    use_onnx_tool: bool = True

    def __init__ (self, network_name: str, onnx_file: PosixPath = None, use_onnx_tool: bool = True) -> None:
        """
        Initialize the ONNX class by loading the ONNX model and extracting the necessary information.
        :param network_name: (String) The name of the ONNX model, to be estimated.
        :param onnx_file: (PosixPath) The ONNX model the user provided for the estimation.
        :param use_onnx_tool: (bool) Whether to use onnx_tool for the calculation of the operations, of a node. (num_ops)
        """
        logger.debug('[__init__]: Start. network_name = %s, use_onnx_tool = %s' % (str(network_name), str(use_onnx_tool)))
        self.network_name = network_name
        self.use_onnx_tool = use_onnx_tool
        
        self.load_onnx_file(onnx_file)
        self.onnx_graph = self.onnx_model.graph
        logger.debug('[__init__]: Successfully loaded the onnx_file.')

        self.load_weights()
        logger.debug('[__init__]: Successfully loaded the weights.')

        if self.use_onnx_tool:
            self.generate_onnx_tool_graph(onnx_file)
            logger.debug('[__init__]: Using onnx_tool. use_onnx_tool = %s' % str(self.use_onnx_tool))

    def load_onnx_file (self, onnx_file: PosixPath) -> None:
        """
        Load the file, provided in onnx_file.
        :param onnx_file: (PosixPath) The ONNX model the user provided for the estimation.
        """
        logger.debug('[load_onnx_file]: Start.')
        if not (os.path.isfile(onnx_file) and os.path.exists(onnx_file)):
            logger.critical('[load_onnx_file]: ONNX file does not exist. onnx_file = %s' % str(os.path.basename(onnx_file)))
            exit(1)
        onnx_model = load(onnx_file)
        logger.debug('[load_onnx_file]: ONNX file has successfully been loaded.')
        if self.use_simplify == True:
            logger.debug('[load_onnx_file]: Simplifying the ONNX file.')
            self.onnx_model, check = simplify(onnx_model)
            assert check, '[load_onnx_file]: Simplified ONNX file could not be validated.'
        else:
            logger.debug('[load_onnx_file]: Not simplifying the ONNX file.')
            self.onnx_model = onnx_model

    def load_weights (self) -> None:
        """
        Extract the weights from the provided ONNX model and store them in a dict.
        """
        logger.debug('[load_weights]: Start.')
        initializer  = self.onnx_graph.initializer
        self.onnx_weights = dict()
        for i in initializer:
            logger.debug('[load_weights]: Extracting initializer. initializer = %s' % str(i.name))
            W = to_array(i)
            self.onnx_weights[i.name] = W
        logger.debug('[load_weights]: Finished loading the weights of the ONNX model.')

    def generate_onnx_tool_graph (self, onnx_file: PosixPath) -> None:
        """
        This method initializes onnx_tool, which is being used to calculate the MACs and FLOPs.
        It's possible, that this initialization fails. In this case we fall back to the regular calculation of ANNETTE.
        One example of the initialization failing is with YOLO v4.
        :param onnx_file: (PosixPath) The ONNX model the user provided for the estimation.
        """
        logger.debug('[generate_onnx_tool_graph]: Start.')
        try:
            m = onnx_tool_model(onnx_file)
            self.onnx_tool_graph = m.graph
            self.onnx_tool_graph.shape_infer()
            self.onnx_tool_graph.profile()
            logger.debug('[generate_onnx_tool_graph]: Successfully initialized onnx_tool with the ONNX model.')
        except:
            self.use_onnx_tool = False
            logger.error('[generate_onnx_tool_graph]: ONNX file could not be imported by onnx_tool.')

    def get_onnx_graph (self) -> GraphProto:
        """
        A method to get the ONNX graph, of the imported ONNX model.
        A description of the GraphProto can be found here: https://onnx.ai/onnx/api/classes.html#onnx.GraphProto
        The graph has been initialized in __init__.
        """
        logger.debug('[get_onnx_graph]: Start.')
        return self.onnx_graph

    def get_node_graph (self) -> RepeatedCompositeFieldContainer:
        """
        A method to get the ONNX node-graph, of the imported ONNX model.
        The graph has been initialized in __init__.
        """
        logger.debug('[get_node_graph]: Start.')
        return self.onnx_graph.node
    
    def get_graph_weights (self, weight_name: str = None) -> ndarray | dict:
        """
        A method to get either all the weights of the imported ONNX model, or a specific one, identified by the name of the initializer.
        The weights have been initialized in load_weights.
        :param weight_name: (String) The name of the weights, which to get.
        """
        logger.debug('[get_graph_weights]: Start.')
        if weight_name:
            logger.debug('[get_graph_weights]: Returning the weights of a specific initializer. weight_name = %s' % str(weight_name))
            return self.onnx_weights[weight_name]
        else:
            logger.debug('[get_graph_weights]: Returning all weights of the ONNX model.')
            return self.onnx_weights
    
    def get_node_weights (self, node: NodeProto) -> ndarray | list:
        """
        A method to get the weights of a specific node.
        :param node: (NodeProto) The node of which to get the weights of.
        """
        logger.debug('[get_node_weights]: Start. node.name = %s' % str(node.name))
        schema = get_schema(node.op_type)
        weight_name = None
        for i, input in enumerate(schema.inputs):
            logger.debug('[get_node_weights]: Searching the weights for the node. input.name = %s' % str(input.name))
            # Since the ONNX model does not differentiate between regular inputs and weights / biases, we try to differentiate them by checking if a weight of this name exists and by checking the definition of the respective node-type.
            # While this has been tested, it's theoretically possible, that ONNX gives the weights of a node another name than "W". In this case, this way of determining the input-weights needs to be adapted.
            if ((input.name.upper() == "W") and (node.input[i] in self.get_graph_weights())):
                logger.debug('[get_node_weights]: The weights have been identified. input.name = %s' % str(input.name))
                weight_name = node.input[i]
        if weight_name:
            logger.debug('[get_node_weights]: The weights have been identified for node. weight_name = %s' % str(weight_name))
            return self.get_graph_weights(weight_name)
        else:
            logger.debug('[get_node_weights]: No weights have been identified for node. weight_name = %s' % str(weight_name))
            return list()

    def get_number_of_node_weights (self, node: NodeProto) -> int:
        """
        A method to get the number of weights, of a specific node.
        :param node: (NodeProto) The node of which to get the number of weights of.
        """
        logger.debug('[get_number_of_node_weights]: Start. node.name = %s' % str(node.name))
        weights = self.get_node_weights(node)
        if isinstance(weights, np.ndarray):
            logger.debug('[get_number_of_node_weights]: The number of weights has been identified for the node. weights.size = %s' % str(weights.size))
            return weights.size
        logger.debug('[get_number_of_node_weights]: The number of weights could not be determined for the node, using onnx_tools. Its possible, that the node does not have any weights.')
        return 0

    def get_tensor_based_on_name (self, tensor_name: str) -> ValueInfoProto | None:
        """
        A method to get a tensor within the ONNX model, based on it's name.
        This method is required to determine the shape of an input.
        :param tensor_name: (String) The tensor-name of which to get the tensor of.
        """
        logger.debug('[get_tensor_based_on_name]: Start. tensor_name = %s' % str(tensor_name))
        return next((x for x in self.onnx_graph.value_info if x.name == tensor_name), None)

    def get_tensor_from_model_input (self, tensor_name: str) -> ValueInfoProto | None:
        """
        A method to get a tensor within the ONNX models inputs, based on it's name.
        Unfortunately, the tensors of the model-inputs and outputs are not stored together with the other tensors.
        Therefore, this method was implemented to also consider the input tensors, in the estimation.
        :param tensor_name: (String) The tensor-name of which to get the tensor of, within the ONNX models inputs.
        """
        logger.debug('[get_tensor_from_model_input]: Start. tensor_name = %s' % str(tensor_name))
        return next((x for x in self.onnx_graph.input if x.name == tensor_name), None)
    
    def get_tensor_from_model_output (self, tensor_name: str) -> ValueInfoProto | None:
        """
        A method to get a tensor within the ONNX models outputs, based on it's name.
        Unfortunately, the tensors of the model-inputs and outputs are not stored together with the other tensors. (in model.graph.value_info)
        Therefore, this method was implemented to also consider the output tensors, in the estimation.
        :param tensor_name: (String) The tensor-name of which to get the tensor of, within the ONNX models outputs.
        """
        logger.debug('[get_tensor_from_model_input]: Start. tensor_name = %s' % str(tensor_name))
        return next((x for x in self.onnx_graph.output if x.name == tensor_name), None)

    def reorder_shape (self, shape: list[int]) -> list[int]:
        """
        The order of input- and output shape of ONNX is [batch-size, number-of-channels, height, width], whereas ANNETTE needs [batch-size, height, width, number-of-channels].
        Therefore, we need to reorder the shapes before using them for the estimation.
        :param shape: (Integer Array) The shape, that should be reordered.
        """
        logger.debug('[reorder_shape]: Start. shape = %s' % str(shape))
        if len(shape) == 4:
            shape[1], shape[2], shape[3] = shape[2], shape[3], shape[1]
            logger.debug('[reorder_shape]: Shape has been reordered. shape = %s' % str(shape))
        logger.debug('[reorder_shape]: Returning the shape. shape = %s' % str(shape))
        return shape
    
    def get_node_inputs (self, node: NodeProto) -> list[str]:
        """
        This method is used as a getter, to get all regular inputs of a node.
        Since ONNX does not differentiate between regular inputs and weights / biases, we need to determine them based on the node-types definition and by checking if an initializer with this name exists.
        If an initializer with the inputs' name exists, it's a weight or a bias.
        :param node: (NodeProto) The node of which to get the regular inputs.
        """
        logger.debug('[get_node_inputs]: Start. node.name = %s' % str(node.name))
        schema = get_schema(node.op_type)
        regular_inputs = list()
        for i, input_name in enumerate(node.input):
            logger.debug('[get_node_inputs]: Searching for the regular inputs of the node. input_name = %s' % str(input_name))
            # Sometimes ONNX defines only one input, if infinitely many tensors can be given as an input, therefore we need to handle the index out-of-bounds (e.g. Concat)
            if (i > len(schema.inputs) - 1):
                schema_input = schema.inputs[len(schema.inputs) - 1]
            else:
                schema_input = schema.inputs[i]
            # Since the ONNX model does not differentiate between regular inputs and weights / biases, we try to differentiate them by checking if a weight of this name exists and by checking the definition of the respective node-type.
            if (not input_name in self.get_graph_weights()) and (schema_input.name.upper() != "W") and (schema_input.name.upper() != "B"):
                logger.debug('[get_node_inputs]: A regular input has been identified for the node. input_name = %s' % str(input_name))
                regular_inputs.append(input_name)
        logger.debug('[get_node_inputs]: The regular inputs have been identified for the node. regular_inputs = %s' % str(regular_inputs))
        return regular_inputs
    
    def get_shape_based_on_tensor_name (self, tensor_name: str) -> None | list[int]:
        """
        This method is a getter to get the shape for a tensor. The tensor is being identified by it's name.
        :param tensor_name: (String) The name of the tensor of which to get which the shape.
        """
        logger.debug('[get_shape_based_on_tensor_name]: Start. tensor_name = %s' % str(tensor_name))
        tensor = self.get_tensor_based_on_name(tensor_name) or self.get_tensor_from_model_input(tensor_name) or self.get_tensor_from_model_output(tensor_name)
        if not tensor:
            logger.warning('[get_shape_based_on_tensor_name]: The tensor was not found. tensor_name = %s' % str(tensor_name))
            return None
        else:
            logger.debug('[get_shape_based_on_tensor_name]: Tensor has been found. tensor = %s' % str(tensor))
            tensor_dimension = tensor.type.tensor_type.shape.dim
            shape = list()
            for dimension_parameter in tensor_dimension:
                logger.debug('[get_shape_based_on_tensor_name]: Iterating through the tensor dimensions to get the shape. dimension_parameter = %s' % str(dimension_parameter))
                if hasattr(dimension_parameter, 'dim_value'):
                    shape.append(dimension_parameter.dim_value)
            logger.debug('[get_shape_based_on_tensor_name]: The shape for the tensor has been found and will be reordered. shape = %s' % str(shape))
            return self.reorder_shape(shape)

    def get_node_input_shape (self, node: NodeProto) -> list[int] | list[list[int]]:
        """
        This method is a getter to get the input shape for a node.
        The difference to the output shape is, that in the case of multiple input shapes, all of them will be returned.
        :param node: (NodeProto) The node of which to get the input shape.
        """
        logger.debug('[get_node_input_shape]: Start. node.name = %s' % str(node.name))
        regular_inputs = self.get_node_inputs(node)
        shapes = list()
        for input in regular_inputs:
            logger.debug('[get_node_input_shape]: Getting the shape for a regular input. input = %s' % str(input))
            shapes.append(self.get_shape_based_on_tensor_name(input))
        if len(shapes) == 0:
            logger.debug('[get_node_input_shape]: No shapes have been found. shapes = %s' % str(shapes))
            return []
        elif len(shapes) == 1:
            logger.debug('[get_node_input_shape]: One shape has been found. shapes = %s' % str(shapes))
            return shapes[0]
        else:
            logger.debug('[get_node_input_shape]: Several shapes have been found, all of which will be returned. shapes = %s' % str(shapes))
            # TODO: Check if all shapes are similar and, if not, add them individually
            return shapes

    def get_node_output_shape (self, node: NodeProto) -> list[int]:
        """
        This method is a getter to get the output shape for a node.
        The difference to the input shape is, that in the case of multiple output shapes, only the first one will be returned.
        :param node: (NodeProto) The node of which to get the input shape.
        """
        logger.debug('[get_node_output_shape]: Start. node.name = %s' % str(node.name))
        outputs = node.output
        shapes = list()
        for output in outputs:
            logger.debug('[get_node_output_shape]: Getting the shape for an output. output = %s' % str(output))
            shapes.append(self.get_shape_based_on_tensor_name(output))
        if len(shapes) == 0:
            logger.debug('[get_node_output_shape]: No shapes have been found. shapes = %s' % str(shapes))
            return []
        elif len(shapes) == 1:
            logger.debug('[get_node_output_shape]: One shape has been found. shapes = %s' % str(shapes))
            return shapes[0]
        else:
            logger.debug('[get_node_output_shape]: Several shapes have been found, of which the first one will be returned. shapes = %s' % str(shapes))
            # TODO: Check if all shapes are similar and, if not, add them individually
            return shapes[0]
    
    def get_node_attributes (self, node: NodeProto) -> dict:
        """
        This method is a getter to get all the attributes of a node and organize them in a dict.
        :param node: (NodeProto) The node of which to get the attributes of.
        """
        logger.debug('[get_node_attributes]: Start. node.name = %s' % str(node.name))
        attributes = dict()
        for attribute in node.attribute:
            logger.info('[get_node_attributes]: An attribute has been found, for the node. attribute.name = %s' % str(attribute.name))
            attributes[attribute.name] = get_attribute_value(attribute)
        logger.debug('[get_node_attributes]: The attributes have been organized in a dict. attributes = %s' % str(attributes))
        return attributes

    def get_node_from_onnx_tools (self, node_name: str) -> None | onnx_tool_node:
        """
        This method is used to get a node from the onnx_tool graph. The onnx_tool graph has been initialized in __init__.
        :param node_name: (String) The name of the node which to get from the onnx_tool graph.
        """
        logger.debug('[get_node_from_onnx_tools]: Start. node_name = %s' % str(node_name))
        if not self.use_onnx_tool:
            logger.debug('[get_node_from_onnx_tools]: The usage of onnx_tool has been disabled.')
            return None
        elif (self.use_onnx_tool) and (node_name in self.onnx_tool_graph.nodemap.keys()):
            logger.info('[get_node_from_onnx_tools]: A node with the searched for name has been found, in the onnx_tools graph.')
            return self.onnx_tool_graph.nodemap[node_name]
        else:
            logger.warning('[get_node_from_onnx_tools]: No node with the searched for name has been found, in the onnx_tools graph.')
            return None

    def get_node_macs_by_name (self, node_name: str) -> int:
        """
        This method is used to get the MACs of a node. If onnx_tools is being used, the MACs are extracted from the onnx_tools graph.
        If onnx_tools is not being used, we return 0. In this case the calculation is done based on the input shape, the output shape and the number of weights.
        This manual calculation happens in the compute_nums method of the respective estimation class. (e.g., /src/annette/estimation/layers/conv.py)
        :param node_name: (String) The name of the node of which to get the MACs.
        """
        logger.debug('[get_node_macs_by_name]: Start. node_name = %s' % str(node_name))
        onnx_tool_node = self.get_node_from_onnx_tools(node_name)
        if onnx_tool_node:
            logger.debug('[get_node_macs_by_name]: A node with the searched for name has been found in the onnx_tools graph.')
            if hasattr(onnx_tool_node, 'macs'):
                logger.info('[get_node_macs_by_name]: The MACs are being extracted from the onnx_tools node. onnx_tool_node.macs[0] = %s' % str(onnx_tool_node.macs[0]))
                return onnx_tool_node.macs[0]
        logger.debug('[get_node_macs_by_name]: No node with the searched for name has been found in the onnx_tools graph. Reverting to the manual calculation of the MACs.')
        return 0

    def get_node_flops_by_name (self, node_name: str) -> int:
        """
        This method is used to get the FLOPs of a node. If onnx_tools is being used, the FLOPs are extracted from the onnx_tools graph.
        If onnx_tools is not being used, we return 0. In this case the calculation is done based on the input shape, the output shape and the number of weights.
        This manual calculation happens in the compute_nums method of the respective estimation class. (e.g., /src/annette/estimation/layers/conv.py)
        Since one FLOP equals two MACs, we need to return the number of MACs and multiply it by two, to get the FLOPs of the node.
        :param node_name: (String) The name of the node of which to get the FLOPs.
        """
        logger.debug('[get_node_flops_by_name]: Start. node_name = %s' % str(node_name))
        return self.get_node_macs_by_name(node_name) * 2 # Times 2, since the operations are MACs, but we need FLOPs

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Basically everything that happens below this comment is used to run the optimization, before the actual estimation.
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def get_node_parent_based_on_input_name (self, input_name: str) -> list[NodeProto]:
        """
        Each node in the graph holds only the information of the names of the inputs.
        Therefore, to get the parent node, we need to check which node has an output whose name is equal to the input of our node.
        :param input_name: (String) The name of the input, of our node, based on which we want to find the parent node.
        """
        logger.debug('[get_node_parent_based_on_input_name]: Start. input_name = %s' % str(input_name))
        onnx_node_graph = self.get_node_graph()
        parent_nodes = list()
        for node in onnx_node_graph:
            if input_name in node.output:
                logger.info('[get_node_parent_based_on_input_name]: Found a parent node. node.name = %s' % str(node.name))
                parent_nodes.append(node)
        logger.debug('[get_node_parent_based_on_input_name]: All parent nodes have been found. len(parent_nodes) = %s' % str(len(parent_nodes)))
        return parent_nodes
    
    def get_node_children_based_on_output_name (self, output_name: str) -> list[NodeProto]:
        """
        Each node in the graph holds only the information of the names of the outputs.
        Therefore, to get the child node, we need to check which node has an input whose name is equal to the output of our node.
        :param output_name: (String) The name of the output, of our node, based on which we want to find the child node.
        """
        logger.debug('[get_node_children_based_on_output_name]: Start. output_name = %s' % str(output_name))
        onnx_node_graph = self.get_node_graph()
        children_nodes = list()
        for node in onnx_node_graph:
            if output_name in node.input:
                logger.debug('[get_node_children_based_on_output_name]: Found a child node. node.name = %s' % str(node.name))
                children_nodes.append(node)
        logger.debug('[get_node_children_based_on_output_name]: All child nodes have been found. len(children_nodes) = %s' % str(len(children_nodes)))
        return children_nodes
    
    def get_node_children_based_on_output_list (self, output_names: list[str]) -> list[NodeProto]:
        """
        This method is just a helper for the get_node_children_based_on_output_name method, to get all children based on multiple output_names.
        In addition to finding the children for each output name, this method also flattens the return value.
        Since every time we call get_node_children_based_on_output_name, we get back an array with all children, we will get something like this: [[...],[...],[...]]
        However, for processing the children, we want the result to look like this: [.......]
        Therefore, this method also flattens the result to put all the children in a one-dimensional array.
        :param output_names: (String Array) The names of the outputs, of our node, based on which we want to find the child nodes.
        """
        logger.debug('[get_node_children_based_on_output_list]: Start. output_names = %s' % str(output_names))
        children_nodes = list()
        for _output in output_names:
            for child in self.get_node_children_based_on_output_name(_output):
                logger.info('[get_node_children_based_on_output_list]: Found a child for the output name. child.name = %s, _output = %s' % (str(child.name), str(_output)))
                children_nodes.append(child)
        logger.debug('[get_node_children_based_on_output_list]: All child nodes have been found. len(children_nodes) = %s' % str(len(children_nodes)))
        return children_nodes

    def node_is_in_graph (self, node: NodeProto) -> bool:
        """
        This method iterates through the ONNX graph and checks, if a node, equal to the node provided in the params, exists in the graph.
        We check if two nodes are equal by comparing their names.
        :param node: (NodeProto) The node provided to check if it exists within the ONNX graph.
        """
        logger.debug('[node_is_in_graph]: Start. node.name = %s' % str(node.name))
        onnx_graph_node = next((x for x in self.get_node_graph() if x.name == node.name), None)
        if onnx_graph_node:
            logger.info('[node_is_in_graph]: A node with the same name has been found in the ONNX graph. onnx_graph_node.name = %s' % str(onnx_graph_node.name))
            return True
        logger.debug('[node_is_in_graph]: No node with the same name has been found in the ONNX graph.')
        return False

    def modify_remove_node (self, node: NodeProto) -> None:
        """
        This method is used to remove a node from the ONNX graph. When removing a node from the ONNX graph, several things need to be done.
        If the node is a leaf node, it's inputs or outputs are probably store in the models inputs and outputs. For that reason we also need to remove it's inputs and outputs from there.
        After this has been done, we can remove the node from the graph. The data structure to store the nodes of the graph is a list. Therefore we can simply use the remove function.
        Additionally, we need to fill the gaps, that are caused by removing an element in a list. The ONNX graph utilizes a singly linked list. Nodes in the list are connected by their outputs and inputs.
        A child node is identified by the fact, that the output of the parent node is the child nodes input. Based on this knowledge, we can fill the gaps. This is done in restore_node_connections_after_modification.
        Lastly, we also want to remove the weights and biases from the model, if the node is not used anymore. This is done in initializer_cleanup_after_modification.
        :param node: (NodeProto) The node we want to remove from the ONNX graph.
        """
        logger.debug('[modify_remove_node]: Start. node.name = %s' % str(node.name))
        if self.node_is_in_graph(node):
            logger.debug('[modify_remove_node]: Node has been found in the ONNX graph. node.name = %s' % str(node.name))
            onnx_graph = self.get_onnx_graph()
            # If one of the inputs was an input of the whole ONNX graph (a leaf node), we also need to remove the input from this list.
            for _input in onnx_graph.input:
                if _input.name in node.input:
                    logger.info('[modify_remove_node]: Removing the nodes inputs from the inputs of the ONNX graph. _input.name = %s' % str(_input.name))
                    onnx_graph.input.remove(_input)
                    break
            # If one of the outputs was an output of the whole ONNX graph (a leaf node), we also need to remove the output from this list.
            for _output in onnx_graph.output:
                if _output.name in node.output:
                    logger.info('[modify_remove_node]: Removing the nodes outputs from the outputs of the ONNX graph. _output.name = %s' % str(_output.name))
                    onnx_graph.output.remove(_output)
                    break
            # The ONNX graph uses a simple list to store it's nodes. Therefore we can use the remove function to remove a node from the graph.
            self.get_node_graph().remove(node)
            logger.debug('[modify_remove_node]: The node has been removed from the ONNX graph.')
            self.restore_node_connections_after_modification(self.get_node_inputs(node), node.output)
            logger.debug('[modify_remove_node]: The connections in the ONNX graph have been restored.')
            # To cleanup initializers (weights / biases) that are not in use anymore, we also remove them from the ONNX model.
            self.initializer_cleanup_after_modification()
            logger.debug('[modify_remove_node]: The initializers have been cleaned up, after a node has been removed.')
        else:
            logger.error('[modify_remove_node]: The node to be removed was not found in the ONNX graph. node.name = %s' % str(node.name))
        
    def restore_node_connections_after_modification (self, node_input: list[str], node_output: RepeatedScalarFieldContainer) -> None:
        """
        For a detailed explaination please see the description of the method modify_remove_node.
        Since the ONNX graph utilizes a singly linked list, removing a node requires the remaining elements of the list to be linked back together.
        In the case of an ONNX graph, this is done by adding the output of the new parent node to the inputs-list of the new child node.
        :param node_input: (String Array) The names of the regular inputs, of the node that was removed.
        :param node_output: (RepeatedScalarFieldContainer) The names of the outputs, of the node that was removed.
        """
        logger.debug('[restore_node_connections_after_modification]: Start. node_input = %s, node_output = %s' % (str(node_input), str(node_output)))
        for _output in node_output:
            for child in self.get_node_children_based_on_output_name(_output):
                output_index = child.input.index(_output)
                child.input.remove(_output)
                logger.info('[restore_node_connections_after_modification]: The output of the removed node has been removed from the inputs of the child node. _output = %s, output_index = %s' % (str(_output), str(output_index)))
                for _input in node_input:
                    # Since we determine regular inputs also by comparing them to the definition of the node type, we need to maintain the order they had originally.
                    # Therefore we insert the new input at the same position the old input has, that has been removed.
                    child.input.insert(output_index, _input)
                    logger.info('[restore_node_connections_after_modification]: The input of the removed node (= the output of the removed nodes parent node) has been added to the inputs of the child node. _input = %s, output_index = %s' % (str(_input), str(output_index)))

    def initializer_cleanup_after_modification (self) -> None:
        """
        This method cleans up initializers (weights / biases) that are not in use anymore.
        It does it by first creating a list of all input names and then iterating through the initializers to remove those, which don't have a corresponding input in the ONNX graph anymore.
        The comparison of initializers and inputs is done based on their names.
        """
        logger.debug('[initializer_cleanup_after_modification]: Start.')
        graph_node_inputs = list()
        onnx_graph = self.get_onnx_graph()
        onnx_node_graph = self.get_node_graph()
        for node in onnx_node_graph:
            for node_input in node.input:
                graph_node_inputs.append(node_input)
        for initializer in onnx_graph.initializer:
            if not (initializer.name in graph_node_inputs):
                logger.info('[initializer_cleanup_after_modification]: Removing an initializer, since no corresponding input was found in the ONNX graph. initializer.name = %s' % str(initializer.name))
                onnx_graph.initializer.remove(initializer)

    def modify_split_node (self, node: NodeProto) -> None:
        """
        This method has not been implemented yet. It's part of the optimization which is done before the estimation of an ONNX model.
        """
        logger.debug('[modify_split_node]: The split modification, of the model optimization, has not been implemented yet. node: ' + str(node.name))

    def modify_merge_node (self, node: NodeProto, optimizer: Optimizer) -> None:
        """
        This method is used to merge two nodes, based on a rule that is defined in the optimizer.
        Merging means in this case, that the primary node (it's defined in the optimizer, which node is the primary node) sustains and it's child node (secondary node), in case
        the rules of the optimizer apply, is being removed. Usually we could take the parameters of the child node and add them to the primary node. However, in the original
        ANNETTE estimation the secondary node is also not considered during the estimation. Hence, we simply remove the secondary node.
        Before a merge, two additional checks can be applied. (conditional check and model optimizer check) Their outputs can determine, whether the node is being merged.
        The usage of these additional checks depends on the definition of the optimizer.
        :param node: (NodeProto) The node, the merge wants to be applied on.
        :optimizer: (Optimizer) The optimizer used, to apply the merge to the node. (see /src/annette/estimation/mapping_model.py)
        """
        logger.debug('[modify_merge_node]: Start. node.name = %s, optimizer.name = %s' % (str(node.name), str(optimizer.name)))
        if self.node_is_in_graph(node):
            logger.debug('[modify_merge_node]: The node has been found on the ONNX graph. node.name = %s' % str(node.name))
            node_output = node.output
            children_nodes = self.get_node_children_based_on_output_list(node_output)
            if len(children_nodes) == 1:
                child_node = children_nodes[0]
                logger.debug('[modify_merge_node]: The node has only one child node, therefore we will continue the merge. child_node.name = %s' % str(child_node.name))
                if child_node.op_type == optimizer.sec_type:
                    logger.debug('[modify_merge_node]: The node type of the child node applies to the rule of the optimizer, therefore we will continue the merge. child_node.op_type = %s, optimizer.sec_type = %s' % (str(child_node.op_type), str(optimizer.sec_type)))
                    merge = True
                    if merge and optimizer.fuse_cond:
                        logger.debug('[modify_merge_node]: Based on the optimizer we start the conditional merge-check. merge = %s, optimizer.name = %s' % (str(merge), str(optimizer.name)))
                        merge = merge and self.modify_merge_conditional(node, child_node, optimizer.fuse_cond)
                        logger.debug('[modify_merge_node]: The state of the merge, after the conditional merge-check, is. merge = %s' % str(merge))
                    if merge and optimizer.est_model and optimizer.conv_dict:
                        logger.debug('[modify_merge_node]: Based on the optimizer we start the model optimization merge-check. merge = %s, optimizer.name = %s' % (str(merge), str(optimizer.name)))
                        merge = merge and self.modify_merge_model_optimizer(node, child_node, optimizer.est_model, optimizer.conv_dict)
                        logger.debug('[modify_merge_node]: The state of the merge, after the model optimization merge-check, is. merge = %s' % str(merge))
                    if merge:
                        logger.debug('[modify_merge_node]: All previous checks were successful. Therefore we start the merging of the node. node.name = %s' % str(node.name))
                        self.modify_merge_simple(node, child_node, optimizer.out_type)
        else:
            logger.error('[modify_merge_node]: The node to be merged was not found in the ONNX graph. node.name = %s' % str(node.name))

    def modify_merge_simple (self, node: NodeProto, child_node: NodeProto, out_type:str) -> None:
        """
        This method is used to merge two nodes. The merge done here is very simple and consists of removed the secondary node.
        For a more detailed description of the merge, please have a look at the description of modify_merge_node.
        For a more detailed description of the removal, please have a look at the description of modify_remove_node.
        :param node: (NodeProto) The primary node, the merge is being applied on.
        :param child_node: (NodeProto) The secondary node, the merge is being applied on. This node is being removed, in the course of the merge.
        :param out_type: (String) The type of the primary node, after the merge. This value is defined by the optimizer.
        """
        logger.info('[modify_merge_simple]: Start. node.name = %s, child_node.name = %s, out_type = %s' % (str(node.name), str(child_node.name), str(out_type)))
        if self.node_is_in_graph(child_node):
            logger.debug('[modify_merge_simple]: We will merge the child node to the primary node. node = %s, child_node = %s' % (str(node.name), str(child_node.name)))
            self.modify_remove_node(child_node)
            logger.debug('[modify_merge_simple]: Removed the child node. child_node = %s' % str(child_node.name))
            # TODO: Macht es hier Sinn die Informationen vom child_node anzuhängen? Sie werden bei der Estimation sowieso nicht berücksichtigt.
            node.op_type = out_type
            logger.debug('[modify_merge_simple]: Changed the type of the primary node. node.op_type = %s' % str(node.op_type))
    
    def modify_merge_conditional (self, node: NodeProto, child_node: NodeProto, merge_condition: dict) -> bool:
        """
        This method is used to check if two nodes should be merged. Several rules can be applied, all of which are defined by the merge conditions, coming from an optimizer.
        For a more detailed description of the merge, please have a look at the description of modify_merge_node.
        :param node: (NodeProto) The primary node, the merge should be applied on.
        :param child_node: (NodeProto) The secondary node, the merge should be applied on.
        :param merge_condition: (dict) The conditions for the primary and the secondary node, which determine if two nodes should be merged or not.
        """
        logger.debug('[modify_merge_conditional]: Start. node.name = %s, child_node.name = %s, merge_condition = %s' % (str(node.name), str(child_node.name), str(merge_condition)))
        def check_merge_condition (condition: dict, attribute_value) -> int:
            # TODO: In future versions of ANNETTE, all of these condition checks could be summarized in one single method.
            logger.debug('[check_merge_condition]: Start. condition[\'cond\'] = %s, condition[\'val\'] = %s, attribute_value = %s' % (str(condition['cond']), str(condition['val']), str(attribute_value)))
            # The following lines check, based on the condition (e.g., == or >) if the attribute value applies to the value defined in the condition.
            if condition['cond'] == "==" and str(attribute_value) == condition['val']:
                return 0
            elif condition['cond'] == ">" and attribute_value > float(condition['val']):
                return 0
            elif condition['cond'] == "<" and attribute_value < float(condition['val']):
                return 0
            elif condition['cond'] == ">=" and attribute_value >= float(condition['val']):
                return 0
            elif condition['cond'] == "<=" and attribute_value <= float(condition['val']):
                return 0
            else:
                return 1

        def perform_condition_check (node: NodeProto, conditions: dict) -> int:
            logger.debug('[perform_condition_check]: Start. node.name = %s, conditions = %s' % (str(node.name), str(conditions)))
            condition_count = 0
            # The following lines basically iterate through all conditions, gets the attribute values and then calls the check in modify_merge_conditional
            for key, condition in conditions.items():
                primary_node = self.compute_nums_for_node(node)
                if not (condition['name'] in primary_node):
                    logger.warning('[perform_condition_check]: Specified attribute was not found on the node. condition[\'name\'] = %s, primary_node = %s' % (str(condition['name']), str(primary_node)))
                    condition_count += 1
                    continue
                attribute = primary_node[condition['name']]
                logger.debug('[perform_condition_check]: Getting the attribute for the condition check. condition[\'name\'] = %s, attribute = %s' % (str(condition['name']), str(attribute)))
                # If a certain index of an attribute should be used, we extract this index-value here.
                if 'i' in condition:
                    if int(condition['i']) <= (len(attribute) - 1):
                        logger.debug('[perform_condition_check]: Getting the specified index of the attribute. condition[\'i\'] = %s, attribute = %s' % (str(condition['i']), str(attribute)))
                        attribute_value = attribute[int(condition['i'])]
                    else:
                        logger.error('[perform_condition_check]: Provided index is out of bounds. Setting the attribute_value to 0. condition[\'i\'] = %s, attribute = %s' % (str(condition['i']), str(attribute)))
                        attribute_value = 0
                else:
                    attribute_value = attribute
                # Based on the extracted values, we start the condition check.
                condition_count += check_merge_condition(condition, attribute_value)
            logger.debug('[perform_condition_check]: All conditions have been evaluated. condition_count = %s' % str(condition_count))
            return condition_count
        
        primary_condition_result = perform_condition_check(node, merge_condition['primary'])
        secondary_condition_result = perform_condition_check(child_node, merge_condition['secondary'])

        logger.info('[modify_merge_conditional]: The condition checks have been conducted. primary_condition_result = %s, secondary_condition_result = %s' % (str(primary_condition_result), str(secondary_condition_result)))

        return (primary_condition_result + secondary_condition_result) == 0

    def compute_nums_for_node (self, node: NodeProto) -> dict:
        """
        This method computes the numbers for the node provided in the parameters.
        Since many numbers are calculated within the compute_nums methods of the respective node-type-classes, we need to implement this here too.
        :param node: (NodeProto) The node which the numbers should be calculated for.
        """
        logger.debug('[compute_nums_for_node]: Start. node.name = %s' % str(node.name))
        node_parameters_dict = dict()
        node_parameters_dict['input_shape'] = self.get_node_input_shape(node)
        node_parameters_dict['output_shape'] = self.get_node_output_shape(node)
        node_parameters_dict['num_weights'] = self.get_number_of_node_weights(node)
        node_parameters_dict['num_ops'] = self.get_node_flops_by_name(node.name)
        attributes = self.get_node_attributes(node)
        for key, attribute_value in attributes.items():
            logger.debug('[compute_nums_for_node]: Processing attribute of the node. node.name = %s, key = %s, attribute_value = %s' % (str(node.name), str(key), str(attribute_value)))
            node_parameters_dict[key] = attribute_value
        # Since some parameters (e.g. the full kernel_shape or the FLOPs) are calculated within compute_nums, we need to call this method.
        if node.op_type in Optimizer.layer_classes:
            logger.debug('[modify_merge_model_optimizer]: Calling the compute_nums method of the node-type class. node.op_type = %s, node_parameters_dict = %s' % (str(node.op_type), str(node_parameters_dict)))
            node_parameters_dict = Optimizer.layer_classes[node.op_type].compute_nums(node_parameters_dict)
        else:
            logger.debug('[modify_merge_model_optimizer]: Calling the compute_nums method of the Base class. node.op_type = %s, node_parameters_dict = %s' % (str(node.op_type), str(node_parameters_dict)))
            node_parameters_dict = Optimizer.layer_classes['Base'].compute_nums(node_parameters_dict)
        return node_parameters_dict

    def modify_merge_model_optimizer (self, node: NodeProto, child_node: NodeProto, estimation_model, convolution_dict: dict) -> bool:
        """
        This method is used to check if two nodes should be merged. The check is based on a model optimizer, provided by the respective ANNETTE optimizer.
        For a more detailed description of the merge, please have a look at the description of modify_merge_node.
        :param node: (NodeProto) The primary node, the merge should be applied on.
        :param child_node: (NodeProto) The secondary node, the merge should be applied on
        :param estimation_model: (Estimation Model) The estimation model used to check if the node should be merged.
        :param convolution_dict: (dict) A dict which defines the vector, that is evaluated by the evaluation model. (see /src/annette/estimation/mapping_model.py)
        """
        logger.debug('[modify_merge_model_optimizer]: Start. node.name = %s, child_node.name = %s' % (str(node.name), str(child_node.name)))
        vector = np.zeros([1, len(convolution_dict)])
        primary_node = self.compute_nums_for_node(node)
        secondary_node = self.compute_nums_for_node(child_node)
        logger.debug('[modify_merge_model_optimizer]: Running the model optimization based on the computed parameters. primary_node = %s, secondary_node = %s' % (str(primary_node), str(secondary_node)))
        # The optimizer defines how the vector should be constructed, that is being used to run the model optimization. This definition is a dict.
        # In the following lines we iterate through this dict and build up the vector with the attributes specified in the optimizer.
        for key, vector_item in convolution_dict.items():
            if isinstance(vector_item, dict):
                # Depending on if the value should be taken from the primary node or from the secondary node, we adapt the node_base here. (again, this is defined in the optimizer)
                if vector_item['layer'] == "primary":
                    logger.debug('[modify_merge_model_optimizer]: Taking the vector item from the primary node. vector_item[\'name\'] = %s' % str(vector_item['name']))
                    node_base = primary_node
                else:
                    logger.debug('[modify_merge_model_optimizer]: Taking the vector item from the secondary node. vector_item[\'name\'] = %s' % str(vector_item['name']))
                    node_base = secondary_node
                # Now we build up the vector and insert the values defined in the optimizer, into the vector. It's possible, that an index of a value should be used, hence the first if-statement.
                if 'i' in vector_item:
                    logger.debug('[modify_merge_model_optimizer]: Taking an index-value of the vector item. vector_item[\'name\'] = %s, vector_item[\'i\'] = %s' % (str(vector_item['name']), vector_item['i']))
                    vector[0, int(key)] = node_base[vector_item['name']][vector_item['i']]
                    logger.debug('[modify_merge_model_optimizer]: Calculated the value of the vector item. node_base[vector_item[\'name\']][vector_item[\'i\']] = %s' % str(node_base[vector_item['name']][vector_item['i']]))
                else:
                    logger.debug('[modify_merge_model_optimizer]: Taking the vector item from the node base. vector_item[\'name\'] = %s' % str(vector_item['name']))
                    vector[0, int(key)] = node_base[vector_item['name']]
                    logger.debug('[modify_merge_model_optimizer]: Calculated the value of the vector item. node_base[vector_item[\'name\']] = %s' % str(node_base[vector_item['name']]))
                if 'dec' in vector_item.keys():
                    logger.debug('[modify_merge_model_optimizer]: Calculating the dec of the vector item. vector_item[\'dec\'] = %s' % str(vector_item['dec']))
                    vector[0, int(key)] = vector[0, int(key)] - vector_item['dec']
                    logger.debug('[modify_merge_model_optimizer]: Calculated the value of the vector item. vector[0, int(key)] - vector_item[\'dec\'] = %s' % str(vector[0, int(key)] - vector_item['dec']))
            else:
                vector[0, int(key)] = vector_item
        logger.debug('[modify_merge_model_optimizer]: Running the model optimization based on the built-up vector. vector = %s' % vector.tostring())
        result = estimation_model.predict(vector)
        logger.debug('[modify_merge_model_optimizer]: Calculated the result of the model optimization. result = %s' % str(result))
        if result == 1.0:
            return True
        else:
            return False
