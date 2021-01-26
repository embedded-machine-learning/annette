import pytest
import sys
sys.path.append("./")
from annette.graph import MMGraph
from annette.graph import AnnetteGraph
from annette.estimation.layer_model import Layer_model 
from annette.estimation.mapping_model import Mapping_model 
from pathlib import Path
import os
import logging

def test_optimization(network="cf_resnet50"):
    model = AnnetteGraph("mobile_original_imagenet","tests/test_data/graph_annette/"+network+".json")

    ncs2_opt = Mapping_model.from_json("tests/test_data/mapping/ov2.json")
    ncs2_opt.run_optimization(model)

    assert True

    return model

def test_estimation(network="cf_resnet50"):
    model = AnnetteGraph("mobile_original_imagenet","tests/test_data/graph_annette/"+network+".json")

    ncs2_opt = Mapping_model.from_json("tests/test_data/mapping/ov2.json")
    ncs2_opt.run_optimization(model)

    # LOAD MODELS
    ncs2_mod = {}
    ncs2_mod['roofline'] = Layer_model.from_json("tests/test_data/generated/ncs2/ncs2-roofline.json")
    ncs2_mod['ref_roofline'] = Layer_model.from_json("tests/test_data/generated/ncs2/ncs2-ref_roofline.json")
    ncs2_mod['statistical'] = Layer_model.from_json("tests/test_data/generated/ncs2/ncs2-statistical.json")
    ncs2_mod['mixed'] = Layer_model.from_json("tests/test_data/generated/ncs2/ncs2-mixed.json")
        
    # APPLY ESTIMATION
    ncs2_res = {}
    ncs2_res['mixed'] = ncs2_mod['mixed'].estimate_model(model)
    ncs2_res['roofline'] = ncs2_mod['roofline'].estimate_model(model)
    ncs2_res['ref_roofline'] = ncs2_mod['ref_roofline'].estimate_model(model)
    ncs2_res['statistical'] = ncs2_mod['statistical'].estimate_model(model)

    assert True

    return model