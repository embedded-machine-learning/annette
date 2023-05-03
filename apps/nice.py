#!/usr/bin/env python3
import asyncio
import os
import time
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager, Queue

from pathlib import Path
import onnx
from onnxsim import simplify, model_info
from annette.estimate import estimate
from annette.m2a import onnx_to_annette
from annette.utils import get_database
from annette.graph import AnnetteGraph
from nicegui import app, events, ui
from plotly import express as px, graph_objects as go
import pandas as pd


pool = ProcessPoolExecutor()

estimation_dataframe = pd.DataFrame(columns=['network', 'mapping', 'hardware', 'res', 'sum', 'time'])


def handle_upload(event: events.UploadEventArguments):
    upload = get_database('graphs', 'onnx', '_tmp.onnx')
    with event.content as f:
        file = open(upload, 'wb')
        for line in f.readlines():
            file.write(line)
        file.close()
    simplify_and_convert()

def convert_to_resolutions(network="_tmp"):
    resolutions = [0.5, 0.75, 1.0, 1.25, 1.5]
    for res in resolutions:
        model_name = f'{network}'
        json_file = get_database('graphs','annette',model_name+'.json')
        annette_graph = AnnetteGraph(model_name, json_file)
        annette_graph.scale_input_resolution(res)
        model_name_res = f'{model_name}_{res}'
        Path.mkdir(get_database('graphs','annette','resolutions'), exist_ok=True)
        annette_graph.to_json(str(get_database('graphs','annette','resolutions',model_name_res+'.json')))

def simplify_and_convert(network="_tmp"):
    filename = get_database('graphs', 'onnx', network+'.onnx')
    # load your predefined ONNX model
    model = onnx.load(str(filename))
    model_name = network
    model_opt, check = simplify(model)

    filename = get_database('graphs', 'onnx', f'{model_name}.onnx')
    onnx.save(model_opt, str(filename))
    
    args = Namespace(network=model_name, inputs='x')
    onnx_to_annette(args)


def compute(network, mapping, layer):
    sum, time = estimate(Namespace(network=network, mapping=mapping, layer=layer))
    return (sum, time)


@ui.page('/')
def main_page():

    plot = None
    def update_chart(sum, network, mapping, layer):
        hw_name = f"{mapping} {layer}"
        # check if network is already in chart
        idx = -1
        if network not in chart.options['xAxis']['categories']:
            chart.options['xAxis']['categories'].append(network)
            # append a zero for all data in the chart
            for i in range(len(chart.options['series'])):
                chart.options['series'][i]['data'].append(0)
        else:
            idx = chart.options['xAxis']['categories'].index(network)
        # get list of all available series names
        series = [s['name'] for s in chart.options['series']]
        idx_hw = -1
        if hw_name not in series:
            chart.options['series'].append({
                'name': hw_name,
                'data': [0] * len(chart.options['xAxis']['categories'])
            })
        else:
            idx_hw = series.index(hw_name)

        # add sum for current network
        chart.options['series'][idx_hw]['data'][idx] = sum
        chart.update()


    async def start_computation(resolution=True):
        global estimation_dataframe
        print(resolution)
        if resolution is True:
            resolutions = [0.5, 0.75, 1.0, 1.25, 1.5]
        else:
            resolutions = [True]
        for res in resolutions:
            if res is True:
                graph = select_annette.value
            else:
                graph = f'resolutions/{select_annette.value}_{res}'
            #remove old plot
            with container_load:
                ui.spinner('dots', size='lg', color='red')
            with container:
                #if last iteration
                if res == resolutions[0]:
                    sum, time = await asyncio.get_event_loop().run_in_executor(pool, compute, graph, select_mapping.value, select_layer.value)
                else:
                    sum, time = compute(graph, select_mapping.value, select_layer.value)
                #make plotly bar chart
                ui.notify(sum)
                # map type of time to colors
                cats = time['type'].unique()
                colors = px.colors.qualitative.Plotly
                color_map = dict(zip(cats, colors))
                time['color'] = time['type'].map(color_map)
                print(time['color'])
                fig = go.Figure(go.Bar(x=time['name'], y=time['time(ms)'],
                                        marker=dict(color = time['color']), textposition='auto', offsetgroup=0
                                    ))
                for c, m in color_map.items():
                    fig.add_trace(go.Bar(
                        x=[0],
                        y=[0],
                        name=c,
                        marker_color=m,
                        offsetgroup=0
                        )
                    )
                fig.update_layout(title=f'Layertimes: {select_annette.value}, {select_mapping.value}, {select_layer.value}')
                # add legend
                fig.update_layout(showlegend = True)
                container_load.clear()
                plot = ui.plotly(fig)
                # fill into dataframe
                if res is True or res == 1.0:
                    res = 1.0
                    update_chart(sum, select_annette.value, select_mapping.value, select_layer.value)
                tmp = {'network': select_annette.value, 'mapping': select_mapping.value, 'hardware': select_layer.value, 'res': res, 'sum': sum, 'time': time}
                estimation_dataframe = estimation_dataframe.append(tmp, ignore_index=True)
        print(resolution)
        print(resolutions)
        print(estimation_dataframe)
        with container_plotly:
            container_plotly.clear()
            fig = px.scatter(estimation_dataframe[estimation_dataframe['network'] == select_annette.value], x='res' , y='sum', color='hardware')
            fig.update_layout(title=f'Estimation: {select_annette.value}')
            fig.update_layout(xaxis_title='Resolution', yaxis_title='time(ms)')
            plot = ui.plotly(fig)

        return (sum, time)

    async def cyclic():
        await start_computation(resolution=True)

    # Create a queue to communicate with the heavy computation process
    queue = Manager().Queue()

    # Get the path to the database folders
    annette_path = get_database('graphs', 'annette')
    layer_path = get_database('models', 'layer')
    mapping_path = get_database('models', 'mapping')
    result_path = get_database('results')

    # get all files in the folder
    def get_files(path):
        l = [f.replace('.json', '') for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.json')]
        l.sort()
        return l
    
    annette_files = get_files(annette_path) 
    layer_files = get_files(layer_path)
    mapping_files = get_files(mapping_path)

    # Create the UI
    ui.markdown('**Upload your onnx file**')
    container_select = ui.row()
    with container_select:
        ui.upload(on_upload=handle_upload)
        select_annette = ui.select(annette_files, value='cf_yolov3')
        select_layer = ui.select(layer_files, value='gap9')
        select_mapping = ui.select(mapping_files, value='simple')
        ui.button('Estimate', on_click=start_computation)
        ui.button('Estimate Resolutions', on_click=cyclic)
        ui.button('Scale Resolution', on_click=(lambda: convert_to_resolutions(select_annette.value)))
        ui.button('Clear', on_click=lambda: container.clear())
    with ui.row() as row:
        chart = ui.chart({
            'title': False,
            'chart': {'type': 'bar'},
            'xAxis': {'categories': []},
            'yAxis': {'title': {'text': 'time(ms)'}},
            'series': [
            ],
        })
    container_load = ui.row().classes('w-full justify-center')
    container_plotly = ui.row()
    container = ui.row()



def get_files(mapping_path):
    return [f.replace('.json', '') for f in os.listdir(mapping_path) if os.path.isfile(os.path.join(mapping_path, f)) and f.endswith('.json')]

def new_func(layer_path):
    return [f.replace('.json', '') for f in os.listdir(layer_path) if os.path.isfile(os.path.join(layer_path, f)) and f.endswith('.json')]



# stop the pool when the app is closed; will not cancel any running tasks
app.on_shutdown(pool.shutdown)

ui.run(port=9999, title='Annette')