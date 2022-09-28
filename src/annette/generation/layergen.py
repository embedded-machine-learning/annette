from __future__ import print_function
from pprint import pprint
from functools import reduce
from annette.estimation import layers
import annette.utils as utils
import json
import numpy as np
import pandas as pd
import pickle as pkl
import copy

from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit

class LayerModelGen():

    def __init__(self, name, layer_type="Base", est_type="roofline", architecture=None):
        self.name = name
        self.layer_type = layer_type
        self.estimation = est_type
        self.op_s = None 
        self.bandwidth = None 
        self.data = None
        self.sweep_data = None

        if architecture:
            self.architecture = architecture 
        else:
            self.architecture = {}
        if not "bit_act" in self.architecture:
            self.architecture["bit_act"] = 8
        if not "bit_weights" in self.architecture:
            self.architecture["bit_weights"] = 8
        self.architecture["f_par"] = self.architecture["c_par"] = self.architecture["h_par"] = self.architecture["w_par"] = 1
        self.architecture["f_alpha"] = self.architecture["c_alpha"] = self.architecture["h_alpha"] = self.architecture["w_alpha"] = 0.0

        self.layer_dict = {}
    
    def read_sweep_data(self, filename):
        self.read_data(filename, sweep=True)

    def read_data(self, filename, sweep=False):
        print("Importing File: %s" % (filename))
        try:
            temp_data = pd.read_pickle(filename)
        except:
            print("Import failed!")
            return None
        
        if sweep is False:
            if self.data is None:
                print("Data imported...")
                self.data = temp_data
            elif all(self.data.columns == temp_data.columns):
                print("Append imported Data...")
                self.data = self.data.append(temp_data, ignore_index=True)
                self.data = self.data.drop_duplicates()
            else:
                print("Data already available and does not fit")
        else:
            if self.sweep_data is None:
                print("Sweep Data imported...")
                self.sweep_data = temp_data
            elif all(self.sweep_data.columns == temp_data.columns):
                print("Append imported Sweep Data...")
                self.sweep_data = self.sweep_data.append(temp_data, ignore_index=True)
                self.sweep_data = self.sweep_data.drop_duplicates()
            else:
                print("Sweep Data already available and does not fit")
        
    def max_op_s(self):
        if self.data is None:
            print("No Data Available")
            return None
        else:
            self.op_s = (self.data['num_ops'] / self.data['time(ms)'] * 1e3).max()
            print("MaxEff P: ", self.op_s)
    
    def max_bandwidth(self):
        if self.data is None:
            print("No Data Available")
            return None
        else:
            print("Architecture: ", self.architecture)
            self.bandwidth = (((self.data['num_inputs'] + self.data['num_outputs'] ) * self.architecture['bit_act']
                + self.data['num_weights'] * self.architecture['bit_weights']) / 8 / self.data['time(ms)'] * 1e3).max()
            print("Bandwidth: ", self.bandwidth)

    def gen_dict(self):

        def set_dict(target, source):
            if hasattr(self, source):
                self.layer_dict[target] = self.__dict__[source]

        set_dict('name', 'name')
        set_dict('layer_type', 'layer_type')
        set_dict('op_s', 'op_s')
        set_dict('bandwidth', 'bandwidth')
        set_dict('architecture', 'architecture')

        set_dict('est_type', 'estimation')
        set_dict('est_dict', 'est_dict')
        set_dict('est_model', 'est_model')
        print(self.layer_dict)

    def generate_roofline(self):
        if self.bandwidth is None:
            self.max_bandwidth()
        if self.op_s is None:
            self.max_op_s()
    
    def generate_architecture(self, fix_params_dict=None):
        if self.sweep_data is None:
            print("No Sweep Data available!")
            return
        print(self.sweep_data)
        go = gen_arc(self.sweep_data, layer_type=self.layer_type, fix_params_dict=fix_params_dict)
        self.architecture.update(go.gen_archarchitecture())

    def generate_estimator(self, est_dict=None, est_model=None, y_val=None, regressor=None, test_size=None):
        """[summary]

        Args:
            est_dict ([type], optional): dictionary containing the features for prediction. Defaults to None.
            est_model ([type], optional): Modelname to store the model. Defaults to None (Not Stored).
            y_val ([type], optional): Value to predict. Defaults to ops/s.
            regressor ([type], optional): sklearn regressor object. Defaults to RandomForrestRegressor.

        Returns:
            [type]: [description]
        """        """"""
        self.generate_roofline()
        self.estimation = 'statistical'
        if est_dict is None:
            est_dict = {'0': 'num_ops',
                '1': 'num_inputs',
                '2': 'num_outputs'}
        self.est_dict = est_dict
        self.layer_dict['est_dict'] = est_dict
        temp = []
        if isinstance(self.est_dict, dict):
            for k, v in self.est_dict.items():
                temp.append(v)
        print(self.est_dict)

        X = self.data[temp].values
        if y_val is None:
            y_val = 'ops/s'
        self.layer_dict['y_val'] = y_val
        y = self.data[y_val].values.reshape(-1,1)

        if not test_size:
            test_size = 0.1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        if regressor is None:
            self.regressor = RandomForestRegressor(min_samples_leaf=1, max_depth=None, n_estimators=50, random_state=False, verbose=False, criterion='mse')
        else:
            self.regressor = regressor

        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)

        self.regressor.fit(X_train[:,:], y_train[:]).score(X_test, y_test) #training the algorithm
        y_pred = self.regressor.predict(X_test)
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)/1e9)
        print('R2 Score:', metrics.r2_score(y_test, y_pred))
        print('Mean Absolute Error Scaled:', metrics.mean_absolute_error(y_test, y_pred)/self.op_s)
        print(f'Mean abs. percentage error: {metrics.mean_absolute_percentage_error(y_test, y_pred) :.2%}')
        print()
        if y_val == 'ops/s':
            # Calculates num_ops / ops_per_sec = secs:
            print('Metrics for seconds:')
            y_test = X_test[:,0] / y_test
            y_pred = X_test[:,0] / y_pred
            print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
            print('R2 Score:', metrics.r2_score(y_test, y_pred))
            print('Mean Absolute Error Scaled:', metrics.mean_absolute_error(y_test, y_pred)/self.op_s)
            print(f'Mean abs. percentage error: {metrics.mean_absolute_percentage_error(y_test, y_pred) :.2%}')

        if est_model is None:
            print("No Estimator stored!\n")
        else:
            self.est_model = est_model
            self.store_model(est_model)
        return y_test, y_pred
    
    def store_model(self, est_model):
        est_model = r'{}'.format(est_model)
        with open(est_model, 'wb') as out_file:
            pkl.dump(self.regressor, out_file)
        print("Estimator stored in "+est_model+ "!\n")
        self.est_model = est_model
        self.layer_dict['est_model'] = est_model
  
    def trans_Conv(self):
        print("Translate prediction parameters to Annette format")
        if self.layer_type in ["Conv", "ConvTranspose2d"]:
            for k, v in self.est_dict.items():
                if v == "height":
                    self.est_dict[k] = {"name": "input_shape", "i": 2}
                if v == "width":
                    self.est_dict[k] = {"name": "input_shape", "i": 1}
                if v == "channels":
                    self.est_dict[k] = {"name": "input_shape", "i": 3}
                if v == "filters":
                    self.est_dict[k] = {"name": "output_shape", "i": 3}
                if v == "k_height":
                    self.est_dict[k] = {"name": "kernel_shape", "i": 1}
                if v == "k_width":
                    self.est_dict[k] = {"name": "kernel_shape", "i": 0}
                if v == "k_stride":
                    self.est_dict[k] = {"name": "strides", "i": 0}
                if v == "k_dilation":
                    self.est_dict[k] = {"name": "dilations", "i": 0}
        elif self.layer_type == "DetphwiseConv":
            for k, v in self.est_dict.items():
                if v == "height":
                    self.est_dict[k] = {"name": "input_shape", "i": 2}
                if v == "width":
                    self.est_dict[k] = {"name": "input_shape", "i": 1}
                if v == "channels":
                    self.est_dict[k] = {"name": "input_shape", "i": 3}
                if v == "k_height":
                    self.est_dict[k] = {"name": "kernel_shape", "i": 1}
                if v == "k_width":
                    self.est_dict[k] = {"name": "kernel_shape", "i": 0}
                if v == "k_stride":
                    self.est_dict[k] = {"name": "strides", "i": 0}
                if v == "k_dilation":
                    self.est_dict[k] = {"name": "dilations", "i": 0}
        elif self.layer_type == "Pool":
            for k, v in self.est_dict.items():
                if v == "height":
                    self.est_dict[k] = {"name": "input_shape", "i": 2}
                if v == "width":
                    self.est_dict[k] = {"name": "input_shape", "i": 1}
                if v == "channels":
                    self.est_dict[k] = {"name": "input_shape", "i": 3}
                if v == "pool_h":
                    self.est_dict[k] = {"name": "kernel_shape", "i": 2}
                if v == "pool_w":
                    self.est_dict[k] = {"name": "kernel_shape", "i": 1}
                if v == "pool_stride":
                    self.est_dict[k] = {"name": "strides", "i": 1}
        elif self.layer_type == "MatMul":
            for k, v in self.est_dict.items():
                if v == "channels":
                    self.est_dict[k] = {"name": "input_shape", "i": 1}
                if v == "filters":
                    self.est_dict[k] = {"name": "output_shape", "i": 1}
        else:
            raise NotImplementedError(f'Not immplemented for layer type: {self.layer_type}')

    def to_json(self):
        tmp = copy.deepcopy(self.layer_dict)
        tmp = utils.bench_to_annette(tmp)
        tmp_json = json.dumps(tmp, indent=4)
        print(tmp_json)
        return tmp_json


class gen_arc():
    def __init__(self, sweep_data, layer_type, fix_params_dict, architecture=None):
        self.sweep_data = sweep_data
        self.architecture = architecture
        self.layer_type = layer_type
        self.fix_params_dict = fix_params_dict

    @staticmethod
    def func3(x, s, alpha, off, speed):
        s = np.floor(s)
        ueff = (x/s) / np.ceil(x/s)
        ueff_alpha = (1-alpha) + ueff*alpha
        return x/ueff_alpha*speed + off + s-np.floor(s)

    @staticmethod
    def loss(xdata, ydata, s, alpha, off, speed):
        s = np.floor(s)
        ueff = (xdata/s) / np.ceil(xdata/s)
        ueff_alpha = (1-alpha) + ueff*alpha

        y = xdata/ueff_alpha*speed + off
        return np.sum((y-ydata) * (y-ydata))

    def get_fix(self, est, *fix_params):
        sweep_cols = list(self.sweep_data.columns)
        f_data = self.sweep_data

        for param in fix_params:
            if param in sweep_cols:
                f_data = f_data[f_data[param] == self.fix_params_dict[param]]

        f_data = f_data.drop_duplicates()

        xdata = f_data[est].to_numpy()
        ydata = f_data['time(ms)'].to_numpy()

        return xdata, ydata, est

    def do_regression(self, data):
        xdata, ydata, est = data
        min_loss = 1000
        for s in range(1, 65):
            popt, pcov = curve_fit(
                lambda xdata, alpha, off, speed: self.func3(xdata, s, alpha, off, speed), xdata, ydata
            )
            lossres = self.loss(xdata, ydata, s, popt[0], popt[1], popt[2])
            if lossres < min_loss:
                min_loss = lossres
                res = {"s":s, "alpha":popt[0], "off":popt[1], "speed":popt[2]}
        res['speed'] = 1/res['speed']
        print(res)

        # Note: There is a logical mismatch between this alpha (alpha_code), vs the one
        # described in the paper (paper_alpha). The following relation holds:
        # alpha_code = (1 - alpha_paper)
        return res['s'], float(np.abs(np.round(res['alpha'], 2)))

    def gen_archarchitecture(self):
        if not self.fix_params_dict:
            self.fix_params_dict = {
                'k_width': 7, 'k_height': 7, 'width': 128, 'height': 128, 'channels': 128, 'filters': 128
            }

        self.architecture = {}

        print(f'Generating architecture for {self.layer_type} layer...')
        if self.layer_type in ['Conv', 'ConvTranspose2d']:
            self.architecture['w_par'], self.architecture['w_alpha'] = \
                self.do_regression(self.get_fix('width','height','channels', 'filters', 'k_height'))
            self.architecture['h_par'], self.architecture['h_alpha'] = \
                self.do_regression(self.get_fix('height','width','channels', 'filters', 'k_height'))
            self.architecture['c_par'], self.architecture['c_alpha'] = \
                self.do_regression(self.get_fix('channels', 'height', 'width', 'filters', 'k_height'))
            self.architecture['f_par'], self.architecture['f_alpha'] = \
                self.do_regression(self.get_fix('filters','channels', 'height', 'width', 'k_height'))
        elif self.layer_type == 'Pool':
            self.architecture['w_par'], self.architecture['w_alpha'] = \
                self.do_regression(self.get_fix('width','height','channels', 'filters', 'pool_h'))
            self.architecture['h_par'], self.architecture['h_alpha'] = \
                self.do_regression(self.get_fix('height','width','channels', 'filters', 'pool_h'))
            self.architecture['c_par'], self.architecture['c_alpha'] = \
                self.do_regression(self.get_fix('channels', 'height', 'width', 'filters', 'pool_h'))
        elif self.layer_type == 'DepthwiseConv':
            self.architecture['w_par'], self.architecture['w_alpha'] = \
                self.do_regression(self.get_fix('width','height','channels', 'k_height'))
            self.architecture['h_par'], self.architecture['h_alpha'] = \
                self.do_regression(self.get_fix('height','width','channels', 'k_height'))
            self.architecture['c_par'], self.architecture['c_alpha'] = \
                self.do_regression(self.get_fix('channels', 'height', 'width', 'k_height'))
        elif self.layer_type == 'MatMul':
            self.architecture['c_par'], self.architecture['c_alpha'] = \
                self.do_regression(self.get_fix('channels', 'filters'))
            self.architecture['f_par'], self.architecture['f_alpha'] = \
                self.do_regression(self.get_fix('filters','channels'))
        else:
            raise NotImplementedError(f'Not implemented for {self.layer_type}')

        return self.architecture
