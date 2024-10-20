# mmdnn-graph-utils

This simple package allows importing mmdnn-ir files, onnx-files and stores them in ANNETTE format Network Architecture description files

## Usage
- as submodule clone to your python module then see *tests* folder for examples, the __init__.py file ensures relative import so no adaptations should be necessary

`import mmdnn-graph-utils as graph`<br>
`print(graph.__dict__)`

to see available classes

## ANNETTE Format

Generally the Annette Format is derived from the MMDNN IR format in terms of attributes, and stored in json format.

### Network header:

    {
        "name" : *network_name*,
        "layers" : {
            "layer_0" : {
                "type" : "Conv",
                "parents" : [],
                "children" : ["layer_1"],
                ...Layer Attributes...
            },
            "layer_1" : {
                "type" : "Relu",
                "parents" : ["layer_0"],
                "children" : [],
                ...Layer Attributes...
            }
        },
        "input_layers" : ["layer_0"],
        "output_layers" : ["layer_1"]
    }


* `"name"` ... denotes the network name and should be identical to the filename
* `"layers"` ... contains a dictionary with all network layers with layer_names as keys and layer attributes as values
* `"input_layers"` ... list of the names of the input_layers
* `"output_layers"` ... list of the namos of the output_layers 

### Layers

* the key of each layer represents the layer name
* the value of each layer contains:
    * `"parents"` ... list of parent layers in the network graph 
    * `"children"` ... list of cild layers in the network graph 
    * `"input_shape"` ... a list of the input shape of the layer inputs
        * NHWC-format
        * if multiple inputs then as a list of lists, order identical to the "parents list" 
    * `"output_shape"` ... a list of the input shape of the layer inputs
        * NHWC-format
        * if multiple outputs then as a list of lists, order identical to the "children list" 
    * `Any additional attribute` ... cild layers in the network graph 

#### Layer Examples 

*DataInput*

    "Placeholder": {
        "type": "DataInput",
        "parents": [],
        "children": [
            "Conv1"
            ],
        "output_shape": [
            1,
            224,
            224,
            3
        ]
    }

*Convolution*

    "resnet_v1_18/conv1/Conv2D": {
        "type": "Conv",
        "parents": [
            "resnet_v1_18/Pad"
        ],
        "children": [
            "resnet_v1_18/conv1/BatchNorm/FusedBatchNorm"
        ],
        "output_shape": [
            -1,
            112,
            112,
            64
        ],
        "input_shape": [
            -1,
            230,
            230,
            3
        ],
        "kernel_shape": [
            7,
            7,
            3,
            64
        ],
        "strides": [
            1,
            2,
            2,
            1
        ],
        "pads": [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0
        ]
    }

for more examples see <a href='./tests/data/'>tests/data/</a>
