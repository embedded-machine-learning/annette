#!/usr/bin/env python3
import asyncio
import os
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from pathlib import Path
import onnx
from onnxsim import simplify
from annette.estimate import estimate
from annette.m2a import onnx_to_annette
from annette.utils import get_database
from annette.graph import AnnetteGraph
from nicegui import app, ui
from plotly import express as px, graph_objects as go
from nicegui.events import UploadEventArguments
import pandas as pd


# Initialize the process pool
pool = ProcessPoolExecutor()

# Global dataframe to store estimations
estimation_dataframe = pd.DataFrame(columns=['network', 'mapping', 'hardware', 'res', 'sum', 'time'])

def handle_upload(event: UploadEventArguments):
    # Handle file upload and save to the appropriate location
    upload = get_database('graphs', 'onnx', '_tmp.onnx')
    with open(upload, 'wb') as file:
        file.write(event.content.read())  # Use the event's content to write to the file
    simplify_and_convert()

def convert_to_resolutions(network="_tmp"):
    # Convert model to different resolutions
    resolutions = [0.5, 0.75, 1.0, 1.25, 1.5]
    for res in resolutions:
        model_name = network
        json_file = get_database('graphs', 'annette', f'{model_name}.json')
        annette_graph = AnnetteGraph(model_name, json_file)
        annette_graph.scale_input_resolution(res)
        model_name_res = f'{model_name}_{res}'
        Path(get_database('graphs', 'annette', 'resolutions')).mkdir(parents=True, exist_ok=True)
        annette_graph.to_json(str(get_database('graphs', 'annette', 'resolutions', f'{model_name_res}.json')))


def simplify_and_convert(network="_tmp"):
    # Simplify the ONNX model and convert it to Annette format
    filename = get_database('graphs', 'onnx', f'{network}.onnx')
    model = onnx.load(str(filename))
    model_name = network
    model_opt, check = simplify(model)

    onnx.save(model_opt, str(get_database('graphs', 'onnx', f'{model_name}.onnx')))
    args = Namespace(network=model_name, inputs='x')
    onnx_to_annette(args)


def compute(network, mapping, layer):
    # Perform estimation
    return estimate(Namespace(network=network, mapping=mapping, layer=layer))

@ui.page('/')
def main_page():
    plot = None

    def update_chart(sum, network, mapping, layer):
        hw_name = f"{mapping} {layer}"
        # Get the series from the dataframe or list
        if network not in estimation_dataframe['network'].unique():
            estimation_dataframe.loc[len(estimation_dataframe)] = {
                'network': network, 'mapping': mapping, 'hardware': layer, 'sum': sum
            }

        # Re-create the Plotly chart with updated data
        fig = go.Figure()
        for hw in estimation_dataframe['hardware'].unique():
            filtered_data = estimation_dataframe[estimation_dataframe['hardware'] == hw]
            fig.add_trace(go.Bar(
                x=filtered_data['network'],
                y=filtered_data['sum'],
                name=hw
            ))
        fig.update_layout(
            title='Estimation Results',
            xaxis_title='Network',
            yaxis_title='Time (ms)',
            barmode='group'
        )
        # Update the plotly chart in the UI
        plot.update(fig)

    async def start_computation(resolution=True):
        global estimation_dataframe
        # Perform computation and chart updating
        resolutions = [0.5, 0.75, 1.0, 1.25, 1.5] if resolution else [True]
        for res in resolutions:
            graph = f'resolutions/{select_annette.value}_{res}' if res != True else select_annette.value
            with container_load:
                ui.spinner('dots', size='lg', color='red')
            with container:
                sum, time = await asyncio.get_event_loop().run_in_executor(pool, compute, graph, select_mapping.value, select_layer.value)
                ui.notify(f'Computation result: {sum}')
                update_chart(sum, select_annette.value, select_mapping.value, select_layer.value)

    async def cyclic():
        await start_computation(resolution=True)

    # Create the UI components
    annette_path = get_database('graphs', 'annette')
    layer_path = get_database('models', 'layer')
    mapping_path = get_database('models', 'mapping')

    annette_files = get_files(annette_path)
    layer_files = get_files(layer_path)
    mapping_files = get_files(mapping_path)

    ui.markdown('**Upload your ONNX file**')
    container_select = ui.row()
    with container_select:
        ui.upload(on_upload=handle_upload)
        select_annette = ui.select(annette_files, value='cf_yolov3')
        select_layer = ui.select(layer_files, value='gap9')
        select_mapping = ui.select(mapping_files, value='simple')
        ui.button('Estimate', on_click=start_computation)
        ui.button('Estimate Resolutions', on_click=cyclic)
        ui.button('Scale Resolution', on_click=lambda: convert_to_resolutions(select_annette.value))
        ui.button('Clear', on_click=lambda: container.clear())

    container_load = ui.row().classes('w-full justify-center')
    container = ui.row()

    # Placeholder for Plotly chart
    with ui.row():
        plot = ui.plotly(go.Figure())  # Initialize with an empty plot


def get_files(path):
    return [f.replace('.json', '') for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.json')]


# Stop the pool when the app is closed
app.on_shutdown(pool.shutdown)

# Run the NiceGUI application
ui.run(port=9999, title='Annette')
