import pytest
import sys
sys.path.append("./")
from pathlib import Path
import logging
import os

import annette.hw_modules.hw_modules.openvino as openvino
import annette.benchmark.generator as generator
import annette.benchmark.matcher as matcher 
from annette import get_database

#logging.basicConfig(level=logging.DEBUG)


__author__ = "Matthias Wess"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"

def run_all(network="annette_bench5",
                bench_config="config_test.csv",
                start=0,
                vis=False,
                match = None,
                hw_name = 'intel_cpu',
                hw_config = 'config_template.yaml'):
    """Run all benchmarks for the given network and configuration.
    args:
        network: str
            Name of the network to be benchmarked
        bench_config: str
            Name of the benchmark configuration file
        start: int
            Start index for the benchmark
        vis: bool
            Visualize the network and the benchmark
        match: dict
            Dictionary with the matchings of the network to the benchmark
        hw_name: str
            Name of the hardware to be used
        hw_config: str
            Name of the hardware configuration file
    """
    openvino_obj = openvino.inference.openvinoClass(
        get_database('configs', hw_config))

    bench1 = matcher.Graph_matcher(network, bench_config, match)
    bench1.run_bench(optimize = openvino_obj.optimize_network, execute = openvino_obj.run_network,
                     parse = openvino.parser.r2a,
                     start = start, vis = vis, hardware = hw_name,
        execute_kwargs = {"report_dir": get_database('benchmarks','tmp'), 'device': 'CPU', 'sleep_time': 0.001})

    try:
        for key, v in bench1.df_out.items():
            print(key)
            print(v)
    except:
        print("No per layer data")
        pass

    assert True


def main():
    run_all(network='annette_bench_conv_padded',bench_config='config_v7_3_sampled.csv', start=0, vis=False)
    run_all(network='annette_bench_conv_padding',bench_config='config_v7_3_sampled.csv', start=0, vis=False)
    run_all(network='annette_bench_conv_padded',bench_config='config_sweep_v2.csv', start=0, vis=False)
    run_all(network='annette_bench_conv_padding',bench_config='config_sweep_v2.csv', start=0, vis=False)
    run_all(network='annette_bench_conv_padded',bench_config='config_v7_3.csv', start=0, vis=False)
    run_all(network='annette_bench_conv_padding',bench_config='config_v7_3.csv', start=0, vis=False)

if __name__ == '__main__':
    main()
