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
    json_file = Path('database','graphs','annette',network+'.json')
    model = AnnetteGraph(network,json_file)

    json_file = Path('database','models','mapping','ov.json')
    ncs2_opt = Mapping_model.from_json(json_file)
    ncs2_opt.run_optimization(model)

    assert True

    return model

def test_estimation(network="cf_resnet50"):
    json_file = Path('database','graphs','annette',network+'.json')
    model = AnnetteGraph(network,json_file)

    json_file = Path('database','models','mapping','ov.json')
    ncs2_opt = Mapping_model.from_json(json_file)
    ncs2_opt.run_optimization(model)

    # LOAD MODELS
    ncs2_mod = {}
    json_file = Path('database','models','layer','ov.json')
    ncs2_mod['roofline'] = Layer_model.from_json("database/models/layer/ncs2-roofline.json")
    ncs2_mod['ref_roofline'] = Layer_model.from_json("database/models/layer/ncs2-ref_roofline.json")
    ncs2_mod['statistical'] = Layer_model.from_json("database/models/layer/ncs2-statistical.json")
    ncs2_mod['mixed'] = Layer_model.from_json("database/models/layer/ncs2-mixed.json")
        
    # APPLY ESTIMATION
    ncs2_res = {}
    ncs2_res['mixed'] = ncs2_mod['mixed'].estimate_model(model)
    ncs2_res['roofline'] = ncs2_mod['roofline'].estimate_model(model)
    ncs2_res['ref_roofline'] = ncs2_mod['ref_roofline'].estimate_model(model)
    ncs2_res['statistical'] = ncs2_mod['statistical'].estimate_model(model)

    assert True

    return model

def test_regression_estimation():
    network_list = ['cf_cityscapes', 'cf_resnet50', 'cf_openpose','tf_mobilenetv1','tf_mobilenetv2']

    json_file = Path('database','models','mapping','ov.json')
    ncs2_opt = Mapping_model.from_json(json_file)

    # LOAD MODELS
    ncs2_mod = {}
    json_file = Path('database','models','layer','ov.json')
    ncs2_mod['roofline'] = Layer_model.from_json("database/models/layer/ncs2-roofline.json")
    ncs2_mod['ref_roofline'] = Layer_model.from_json("database/models/layer/ncs2-ref_roofline.json")
    ncs2_mod['statistical'] = Layer_model.from_json("database/models/layer/ncs2-statistical.json")
    ncs2_mod['mixed'] = Layer_model.from_json("database/models/layer/ncs2-mixed.json")
        
    for network in network_list:
        json_file = Path('database','graphs','annette',network+'.json')
        model = AnnetteGraph(network,json_file)
        ncs2_opt.run_optimization(model)
        # APPLY ESTIMATION
        ncs2_res = {}
        ncs2_res['mixed'] = ncs2_mod['mixed'].estimate_model(model)
        ncs2_res['roofline'] = ncs2_mod['roofline'].estimate_model(model)
        ncs2_res['ref_roofline'] = ncs2_mod['ref_roofline'].estimate_model(model)
        ncs2_res['statistical'] = ncs2_mod['statistical'].estimate_model(model)

    assert True
