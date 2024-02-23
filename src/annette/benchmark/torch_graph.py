import torch as t
from torch import nn

class TorchGraphNode(nn.Module):
	def __init__(self, operation, parents):
		super(TorchGraphNode, self).__init__()
		self.operation = operation
		self.parents = parents
		self.result = None

	def forward(self, inputs):
		return self.operation(*inputs)

class TorchGraph(nn.Module):
	def __init__(self, name=None):
		super(TorchGraph, self).__init__()
		self.name = name
		self.output_layer_names = []

	def get_node(self, name):
		return self.get_submodule(name)

	def add_node(self, name, operation, parents=[], is_output=False):
		assert type(parents) in [list, str]
		if type(parents) == str:
			parents = [parents]

		self.add_module(name, TorchGraphNode(operation, parents))

		if is_output:
			self.output_layer_names.append(name)

	def forward(self, inputs):
		if type(inputs) == t.Tensor:
			inputs = [inputs]

		for _, node in self.named_children():
			if node.parents:
				operands = [self.get_node(p).result for p in node.parents]
			else: # no parents => is input node
				operands = inputs

			node.result = node(operands)

		return tuple([self.get_node(o).result for o in self.output_layer_names])

if __name__ == '__main__':
	graph = TorchGraph(name='sample_pt_graph')
	graph.add_node('conv0', nn.Conv2d(3, 3, 3))
	graph.add_node('conv1', nn.Conv2d(3, 3, 3))
	graph.add_node('relu0', nn.ReLU(), parents='conv0')
	graph.add_node('relu1', nn.ReLU(), parents=['conv1'])
	graph.add_node('add0', t.add, parents=['relu0', 'relu1'], is_output=True)

	inp = t.rand(1, 3, 42, 42)

	graph_output = graph(inp)
	print(graph_output)

	# export to .pt file:
	graph.eval()
	t.save(graph, graph.name + '.pt')

	# export to onnx:
	graph.eval()
	t.onnx.export(graph, inp, graph.name + '.onnx')

