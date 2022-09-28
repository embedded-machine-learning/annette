import logging

from .annette_graph import *
try:
    from .mmdnn_graph import *
except ModuleNotFoundError:
    logging.error("Warning: MMDnnmodule could not be imported! To use, install mmdnn!")
try:
    from .onnx_graph import *
except ModuleNotFoundError:
    logging.error("ONNX module could not be imported! To use, install onnx!")