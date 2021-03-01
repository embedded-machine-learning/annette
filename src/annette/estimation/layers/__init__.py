from __future__ import absolute_import

from annette.estimation.layers.base import BaseLayer
from annette.estimation.layers.conv import ConvLayer 
from annette.estimation.layers.convtranspose import ConvTransposeLayer 
from annette.estimation.layers.depthwiseconv import DepthwiseConvLayer 
from annette.estimation.layers.pool import PoolLayer 
from annette.estimation.layers.convpool import ConvPoolLayer 
from annette.estimation.layers.add  import AdditionLayer 
from annette.estimation.layers.fc   import FullyConnectedLayer
from annette.estimation.layers.input import InputLayer

# Layer Dictionary, Naming has to be equal to the Annette Naming convention
__layer_classes__ = {
    'Base': BaseLayer,
    'Conv': ConvLayer,
    'ConvTranspose': ConvTransposeLayer,
    'DepthwiseConv': DepthwiseConvLayer,
    'Pool': PoolLayer,
    'ConvPool': ConvPoolLayer,
    'Add': AdditionLayer,
    'DataInput': InputLayer,
    'FullyConnected':FullyConnectedLayer,
}
