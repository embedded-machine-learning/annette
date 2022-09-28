
from __future__ import print_function
from pprint import pprint
from functools import reduce
from annette.estimation import layers
import json
import numpy as np
import pandas as pd
import pickle as pkl

from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

class MappingModelGen():
    def __init__(self, name, prim_type, sec_type, out_type):
        self.name = name
        self.prim_type = prim_type
        self.sec_type = sec_type 
        self.out_type = out_type 

        self.data = None

        self.desc = {}

    def read_data(self, filename):
        print("Importing File: %s" % (filename))
        try:
            temp_data = pd.read_pickle(filename)
        except:
            print("Import failed!")
            return None

        if self.data is None:
            print("Data imported...")
            self.data = temp_data
        elif all(self.data.columns == temp_data.columns):
            print("Append imported Data...")
            self.data = self.data.append(temp_data, ignore_index = True) 
            self.data = self.data.drop_duplicates()
        else:
            print("Data already available and does not fit")
        
    def gen_dict(self):

        def set_dict(target, source):
            print("checktdict")
            if hasattr(self, source):
                self.desc[target] = self.__dict__[source]
                print("dict exitsts")
        set_dict('name', 'name')
        set_dict('prim_type', 'prim_type')
        set_dict('sec_type', 'sec_type')
        set_dict('out_type', 'out_type')
        set_dict('fuse_cond', 'fuse_cond')
        set_dict('est_model', 'est_model')

        print(self.desc)

    def generate_estimator(self, est_dict=None, est_model=None, y_val=None, classifier=None):
        """Generates statistical estimation function based on assigned estimation data"""
        self.estimation = 'statistical'
        if est_dict is None:
            est_dict = {'0': 'num_ops',
                '1': 'num_inputs',
                '2': 'num_outputs'}
        self.est_dict = est_dict
        self.desc['conv_dict'] = est_dict
        temp = []
        if isinstance(self.est_dict, dict):
            for k, v in self.est_dict.items():
                temp.append(v)
        print(self.est_dict)
        X = self.data[temp].values
        if y_val is None:
            y_val = 'fused'
        y = self.data[y_val].values.reshape(-1,1).astype('float')   #wrong datatype?
        print(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        if classifier is None:
            self.classifier = DecisionTreeClassifier(random_state=0,  max_depth=None, ccp_alpha = 0.001)
        else:
            self.classifier = classifier 
        
        self.classifier.fit(X_train[:,:], y_train[:]).score(X_test, y_test) #training the algorithm
        y_pred = self.classifier.predict(X_test)
        y_train_pred = self.classifier.predict(X_train)

        print(classification_report(y_test, y_pred))
        print(classification_report(y_train, y_train_pred))

        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
        print('R2 Score:', metrics.r2_score(y_test, y_pred))
        if est_model is None:
            print("Estimator not stored!\n")
        else:
            self.est_model = est_model
            self.store_model(est_model)
        return y_test, y_pred
    
    def store_model(self, est_model):
        est_model = r'{}'.format(est_model)
        pkl.dump(self.classifier, open(est_model, 'wb'))
        print("Estimator stored in "+est_model+ "!\n")
        self.est_model = est_model
    
    def add_fuse_condition(self, fuse_cond):
        self.fuse_cond = fuse_cond
        self.gen_dict()

  
    def trans_Conv(self):
        print("translate convolution parameters")
        if self.prim_type == "Conv":
            target = "primary"
        elif  self.sec_type == "Conv":
            target = "secondary"
        else:
            None

        if target:
            for k, v in self.est_dict.items():
                if "name" not in self.est_dict[k]:
                    self.est_dict[k] = {"name": self.est_dict[k]}

                if v == "height":
                    self.est_dict[k] = {"name": "input_shape", "i": 1}
                if v == "width":
                    self.est_dict[k] = {"name": "input_shape", "i": 2}
                if v == "channels":
                    self.est_dict[k] = {"name": "input_shape", "i": 3}
                if v == "filters":
                    self.est_dict[k] = {"name": "output_shape", "i": 3}
                if v == "k_height":
                    self.est_dict[k] = {"name": "kernel_shape", "i": 0}
                if v == "k_width":
                    self.est_dict[k] = {"name": "kernel_shape", "i": 1}

                self.est_dict[k]["layer"] = target

    def to_json(self):
        return json.dumps(self.desc, indent=4)