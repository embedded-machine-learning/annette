#!/usr/bin/env python3
from flask import Flask, request, jsonify, abort
from werkzeug.utils import secure_filename
from os import listdir, walk
from os.path import isfile, join
from pathlib import Path
from datetime import datetime

from annette import get_database
from annette.estimate import estimate_onnx
from annette.graph import ONNX

import json
import csv

UPLOAD_FOLDER = get_database('graphs', 'onnx')
ALLOWED_EXTENSIONS = {'onnx'}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024 # 500MB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

results = list()
layer_analysis = dict()

### Helper Functions ###

def list_all_files_in_directory (path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    return sorted(files)

def list_all_files_in_all_subdirectories (path, extension):
    result = []
    for dirpath, dirnames, filenames in walk(path):
        result.extend([join(dirpath, filename) for filename in filenames if filename.endswith(extension)])
    return result

def list_all_layer_model_details (file_paths):
    layer_models = list()
    for layer_model in file_paths:
        with open(get_database('models', 'layer') / layer_model, 'r') as file:
            file_data = json.load(file)
            layer_models.append({
                'name': layer_model,
                'hardware_description': file_data.get('hardware_description', None)
            })
    return layer_models

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_results ():
    result_files = list_all_files_in_all_subdirectories(get_database('results'), '.json')
    for result in result_files:
        with open(result, 'r') as file:
            file_data = json.load(file)
            results.append({
                'sum': file_data['sum'],
                'hardware': file_data['hardware'],
                'network': file_data['network'],
                'mapping': file_data['model']
            })

def initialize_layer_analysis ():
    result_files = list_all_files_in_all_subdirectories(get_database('results'), '.csv')
    for result in result_files:
        with open(result, 'r') as file:
            file_data = csv.DictReader(file, delimiter=',')
            for line in file_data:
                node_type = line.get('estimation_type') or line.get('type')
                if node_type is None:
                    continue
                if not (node_type in layer_analysis):
                    layer_analysis[node_type] = list()
                layer_analysis[node_type].append({
                    'result': Path(file.name).parent.name + '_' + Path(file.name).stem,
                    'estimate': float(line['time(ms)'])
                })

### Server Initialization ###

with app.app_context():
    initialize_results()
    initialize_layer_analysis()

### Route Definitions ###

@app.route('/networks')
def get_all_networks ():
    return jsonify(list_all_files_in_directory(get_database('graphs', 'onnx')))

@app.route('/layer-models')
def get_all_hardware_platforms ():
    layer_model_files = list_all_files_in_directory(get_database('models', 'layer'))
    return jsonify(list_all_layer_model_details(layer_model_files))

@app.route('/mapping-models')
def get_all_mapping_models ():
    return jsonify(list_all_files_in_directory(get_database('models', 'mapping')))

@app.route('/estimate')
def run_onnx_estimation ():
    argument_object = lambda: None
    argument_object.network = request.args.get('network')
    argument_object.layer = request.args.get('layer')
    argument_object.mapping = request.args.get('mapping')
    argument_object.onnx_tool = request.args.get('onnx_tool') or True
    argument_object.save_optimized_model = request.args.get('save_optimized_model') or False
    
    # TODO: Check if estimation already exists and return the existing result
    try:
        result_tuple = estimate_onnx(argument_object)
    except Exception as e:
        abort(500, e)

    response = dict()
    response['estimation'] = result_tuple[0]
    response['layerResult'] = json.loads(result_tuple[1].to_json(orient='records'))

    return jsonify(response)

@app.route('/results')
def get_results ():
    request_network = request.args.get('network')
    request_layer = request.args.get('layer')
    if not request_network and not request_layer:
        return sorted(results, key=lambda x: x['sum'])
    elif request_network and not request_layer:
        return sorted([x for x in results if x['network'] == request_network], key=lambda x: x['sum'])
    elif not request_network and request_layer:
        return sorted([x for x in results if x['hardware'] == request_layer], key=lambda x: x['sum'])
    else:
        abort(500)

@app.route('/layer-analysis')
def get_layer_analysis ():
    request_type = request.args.get('type')
    if request_type:
        if not (request_type in layer_analysis):
            return jsonify(list())
        else:
            return jsonify(layer_analysis[request_type])
    else:
        average_layer_analysis = dict()
        for node_type, value in layer_analysis.items():
            result_list = [x['estimate'] for x in value]
            average_layer_analysis[node_type] = {
                'average': sum(result_list) / len(result_list),
                'count': len(result_list),
                'details': value
            }
        return jsonify(average_layer_analysis)

@app.route('/upload', methods=['POST'])
def upload_network_onnx_file():
    if 'file' not in request.files:
        # No file provided
        abort(500)
        return
    file = request.files['file']
    if file.filename == '':
        # Invalid filename
        abort(500)
        return
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = join(app.config['UPLOAD_FOLDER'], filename)
        if isfile(filepath):
            return jsonify({'error': 'File already exists'}), 409
        file.save(filepath)
        return 'Success!'
    else:
        abort(500)

@app.route('/network-graph')
def get_network_graph ():
    network_name = request.args.get('network')
    if not network_name:
        abort(500)
        return
    flowchart = 'flowchart TD\n'
    onnx_file = get_database('graphs', 'onnx', network_name + '.onnx')
    onnx_model = ONNX(network_name, onnx_file, False)
    for node in onnx_model.get_node_graph():
        regular_inputs = onnx_model.get_node_inputs(node)
        for input in regular_inputs:
            parents = onnx_model.get_node_parent_based_on_input_name(input)
            for parent in parents:
                flowchart += ('%s(%s) --> %s(%s)\n' % (parent.name, parent.op_type, node.name, node.op_type))
        # Unfortunately, clicking on a node inside the graph does not work as of now. Issue: See also: https://github.com/dword-design/vue-mermaid-string/issues/197 and https://github.com/mermaid-js/mermaid/issues/4346
        # flowchart += ('click %s\n' % (node.name))
    return flowchart

@app.route('/feedback', methods=['POST'])
def save_feedback():
    feedback = request.form.get('feedback')
    if not feedback:
        abort(500)
        return
    else:
        date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        feedback_file = open(get_database('feedback.txt'), 'a')
        feedback_file.write('%s:\n%s\n\n' % (str(date), str(feedback)))
        feedback_file.close()
        return 'Success!'