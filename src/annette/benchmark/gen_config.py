from copy import deepcopy
import sys
from annette import get_database
import numpy as np
import pandas as pd
import pickle


def calc_total_configpoints(dimensions):
    return np.prod([len(d) for d in dimensions])


def aiware_guess_runtime(config_points):
    print(
        f'AiWare guessed benchmark time: {0.008 * config_points / 60} hours.')


def guess_runtime(config_points, time_per_config=1):
    print(
        f'AiWare guessed benchmark time: {time_per_config * config_points / 60 / 60} hours.')


def range_increase_power2(start, end, exponent_start, elem_per_exp=2):
    out = [start]
    next_elem = start
    exponent = exponent_start
    while next_elem < end:
        add_elem = 2**exponent
        for i in range(elem_per_exp):
            next_elem += add_elem
            if next_elem < end:
                out.append(next_elem)
        exponent += 1

    return out


if __name__ == '__main__':
    WIDTH = range_increase_power2(8, 641, 2)  # range(8, 1025, 16)
    HEIGHT = range_increase_power2(8, 641, 2)  # range(8, 1025, 16)
    CHANNELS = [3] + range_increase_power2(8, 2049, 2)  # range(16, 1025, 16)
    # FILTERS = range(1) #[1] + range_increase_power2(8, 2049, 2)
    # K_SIZE = range(10, 11, 2)
    # STRIDE = range(1, 2, 1)
    # DILATION = range(1, 2, 1)
    # BATCH_SIZE = range(1, 2, 1)

    # WIDTH = [32*(i+1) for i in range(25)]
    # WIDTH = [32*(i+1) for i in range(25)]
    # HEIGHT = [8]
    CHANNELS = [3] + range_increase_power2(8, 1026, 2)  # range(16, 1025, 16)
    # CHANNELS = [3, 16, 32, 48]
    # range(16, 1025, 16)
    FILTERS = [65] + range_increase_power2(80, 2049, 4, 4)
    # CHANNELS = [3] + [16*i for i in range(1, 100 )]
    # FILTERS = [64 + 16*i for i in range(100)]
    K_SIZE = range(3, 8, 2)
    STRIDE = range(1, 2, 1)
    DILATION = range(1, 2, 1)
    BATCH_SIZE = range(1, 2, 1)

    # WIDTH = [32, 224]
    # HEIGHT = [8]
    # CHANNELS = [3] + range_increase_power2(8, 2049, 2, 4) #range(16, 1025, 16)
    # CHANNELS = [3, 16, 128, 256]
    # FILTERS = [3, 16, 128, 256] #range(16, 1025, 16)
    # CHANNELS = [3] + [16*i for i in range(1, 100 )]
    # FILTERS = [64 + 16*i for i in range(100)]
    K_SIZE = [1, 3, 5]
    STRIDE = [1]
    DILATION = [1]
    BATCH_SIZE = [1]

    FILTERS = CHANNELS

    print(CHANNELS)
    print(FILTERS)
    tot_points = calc_total_configpoints(
        [WIDTH, HEIGHT, CHANNELS, FILTERS, K_SIZE, STRIDE, DILATION, BATCH_SIZE]
    )
    # print(f'Total config points: {tot_points}')
    # sys.exit()

    aspect_limit = 2
    input_limit = 1024*1024*16
    output_limit = 1024*1024*16

    points = []
    for w in WIDTH:
        for h in HEIGHT:
            if w/aspect_limit > h or h/aspect_limit > w:
                continue
            for c in CHANNELS:
                for f in FILTERS:
                    # for f in [c]:
                    if w*h*c > input_limit or w*h*f > output_limit:
                        continue
                # for f in [2*i*c for i in [1, 2] if 2*i*c < 257]:
                    for k in K_SIZE:
                        for s in STRIDE:
                            for d in DILATION:
                                for b in BATCH_SIZE:
                                    point = (w, h, c, f, k, s, d, b)
                                    points.append(point)

    config = pd.DataFrame(
        points,
        columns=['width', 'height', 'channels', 'filters',
                 'k_size', 'stride', 'dilation', 'batch_size']
    )

    print(f'Total config points: {len(config)}')

    # config['stride'] = config['k_size']
    # config['filters'] = config['channels']
    # config = config[config['filters'] % 16 == 0]
    # config = config[config['width'] % 8 == 0]
    # config = config[config['height'] % 8 == 0]
    # config = config[config['height']  == config['width']]
    config.drop_duplicates()
    print(config)
    config.to_csv(get_database('benchmarks', 'config',
                  'config_v7_3.csv'), index=False)

    guess_runtime(len(config), 5)
    # with open('all_config_combinations.p', 'wb') as f:
    #     pickle.dump(config, f)
