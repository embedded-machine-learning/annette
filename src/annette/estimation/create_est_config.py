import numpy as np
import pandas as pd

def gen_config_conv2d(num_config_points=100):
    config_points = []
    channel_values = [3] + [8*i for i in range(1, 33)]

    for _ in range(num_config_points):
        dp = {}

        # dp['width'] = dp ['height'] = np.random.randint(32, 1025)
        rand_size = np.random.randint(32, 1025)
        dp['width'] = dp['height'] = rand_size if rand_size % 2 == 0 else rand_size + 1

        # dp['channels'] = np.random.randint(3, 33)
        # dp['filters'] = np.random.randint(dp['channels'], 65)

        dp['channels'] = channel_values[np.random.randint(0, len(channel_values))]
        dp['filters'] = channel_values[np.random.randint(0, len(channel_values))]

        k_size = np.random.randint(1, 8)
        dp['k_size'] = k_size if k_size % 2 == 1 else k_size - 1
        dp['stride'] = 1
        dp['dilation'] = 1
        dp['batch_size'] = 1
        config_points.append(dp)

    return config_points

def gen_config_convtranspose2d(num_config_points=100):
    config_points = []
    channel_values = [3] + [8*i for i in range(1, 33)]

    for _ in range(num_config_points):
        dp = {}

        # dp['width'] = dp ['height'] = np.random.randint(32, 1025)
        rand_size = np.random.randint(32, 512)
        dp['width'] = dp['height'] = rand_size if rand_size % 2 == 0 else rand_size + 1

        # dp['channels'] = np.random.randint(3, 33)
        # dp['filters'] = np.random.randint(dp['channels'], 65)

        dp['channels'] = channel_values[np.random.randint(0, len(channel_values))]
        dp['filters'] = channel_values[np.random.randint(0, len(channel_values))]

        k_size = np.random.randint(1, 8)
        dp['k_size'] = k_size if k_size % 2 == 1 else k_size - 1
        dp['stride'] = 1
        dp['dilation'] = 1
        dp['batch_size'] = 1
        config_points.append(dp)

    return config_points

def gen_config_dwconv2d(num_config_points=100):
    config_points = []
    channel_values = [3] + [8*i for i in range(1, 65)]

    for _ in range(num_config_points):
        dp = {}

        rand_size = np.random.randint(32, 1025)
        dp['width'] = dp['height'] = rand_size if rand_size % 2 == 0 else rand_size + 1

        dp['channels'] = channel_values[np.random.randint(0, len(channel_values))]
        dp['filters'] = 1

        k_size = np.random.randint(3, 8)
        dp['k_size'] = k_size if k_size % 2 == 1 else k_size - 1
        dp['stride'] = 1
        dp['dilation'] = 1
        dp['batch_size'] = 1
        config_points.append(dp)

    return config_points

def gen_config_fc(num_config_points=100):
    config_points = []
    channel_values = [4] + [8*i for i in range(1, 129)]

    for _ in range(num_config_points):
        dp = {}
        dp['width'] = dp['height'] = 1
        dp['channels'] = channel_values[np.random.randint(0, len(channel_values))]
        dp['filters'] = channel_values[np.random.randint(0, len(channel_values))]

        config_points.append(dp)

    return config_points

def gen_config_pool(num_config_points=100):
    config_points = []
    channel_values = [3] + [8*i for i in range(1, 65)]

    for _ in range(num_config_points):
        dp = {}

        rand_size = np.random.randint(32, 1025)
        dp['width'] = dp['height'] = rand_size if rand_size % 2 == 0 else rand_size + 1

        dp['channels'] = channel_values[np.random.randint(0, len(channel_values))]
        dp['filters'] = dp['channels']

        dp['k_size'] = [2, 4, 8][np.random.randint(0, 3)]
        dp['stride'] = 2
        dp['dilation'] = 1
        dp['batch_size'] = 1
        config_points.append(dp)

    return config_points


if __name__ == '__main__':
    CONF_FILENAME = 'compare_conv-transpose2d_regular_new.csv'

    # conf_points = gen_config_conv2d(100)
    conf_points = gen_config_convtranspose2d(100)
    # conf_points = gen_config_dwconv2d(100)
    # conf_points = gen_config_fc(100)
    # conf_points = gen_config_pool(100)

    conf_df = pd.DataFrame(conf_points)
    print(conf_df)
    conf_df.to_csv(CONF_FILENAME, index=False)
