import numpy as np
import pandas as pd

def gen_config_conv2d(param_range=256, fixed_val=128):
    config_points = []
    params = {'width', 'height', 'channels', 'filters'}

    for p in params:
        other_params = params - {p}
        for i in range(1, param_range+1):
            dp = {}
            dp[p] = i
            for o in other_params:
                dp[o] = fixed_val
            dp['k_size'] = 3
            dp['stride'] = 1
            dp['dilation'] = 1
            dp['batch_size'] = 1
            config_points.append(dp)

    return config_points

def gen_config_dwconv2d(param_range=256, fixed_val=128):
    config_points = []
    params = {'width', 'height', 'channels'}

    for p in params:
        other_params = params - {p}
        for i in range(1, param_range+1):
            dp = {}
            dp[p] = i
            for o in other_params:
                dp[o] = fixed_val
            dp['filters'] = 1
            dp['k_size'] = 7
            dp['stride'] = 1
            dp['dilation'] = 1
            dp['batch_size'] = 1
            config_points.append(dp)

    return config_points

def gen_config_fc(param_range=256, fixed_val=128):
    config_points = []
    params = {'channels', 'filters'}

    for p in params:
        other_params = params - {p}
        for i in range(1, param_range+1):
            dp = {}
            dp['width'] = dp['height'] = 1
            dp[p] = i
            for o in other_params:
                dp[o] = fixed_val

            config_points.append(dp)

    return config_points

def gen_config_pool(param_range=256, fixed_val=128):
    config_points = []
    params = {'width', 'height', 'channels'}

    for p in params:
        other_params = params - {p}
        for i in range(2, param_range+1):
            dp = {}
            dp[p] = i
            for o in other_params:
                dp[o] = fixed_val

            dp['filters'] = dp['channels']
            dp['k_size'] = 2
            dp['stride'] = 2
            dp['dilation'] = 1
            dp['batch_size'] = 1

            config_points.append(dp)

    return config_points


if __name__ == '__main__':
    CONF_FILENAME = 'conv2d_finesweep.csv'

    conf_points = gen_config_conv2d(256, 32)
    # conf_points = gen_config_fc(256, 128)
    # conf_points = gen_config_pool(256, 128)
    # conf_points = gen_config_dwconv2d(256, 128)

    conf_df = pd.DataFrame(conf_points)
    print(conf_df)
    conf_df.to_csv(CONF_FILENAME, index=False)
