from __future__ import division, print_function, absolute_import

import numpy as np
import pandas as pd
import tensorflow as tf

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = {"workclass": 10, "education": 17, "marital_status":8, 
                       "occupation": 16, "relationship": 7, "race": 6, 
                       "gender": 3, "native_country": 43, "age_binned": 14}
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]

INDICATOR_LIST = []
INDICATOR_PARAM_LIST = []
INDICATOR_PARAM_ITER_PERIOD = [2,3,5,8,13,21,34,55,89,144]

BLANK_LIST = ['','','','','','','','','','','','','','']

OPTIMIZER_LIST = ['Optimizer','SGD','RMSProp','Adam','Momentum','AdaGrad','Ftrl','AdaDelta'] # 8 types
ACTIVATION_LIST = ['linear','tanh','sigmoid','softmax','softplus','softsign','relu','relu6','leaky_relu','prelu','elu'] # 11 types
LAYER_BASIC_LIST = ['fully_connected'] # Only 'fully_connected' needs activation
LAYER_CORE_LIST = ['dropout','highway']
LAYER_RECURRENT_LIST = ['lstm','bidirectional_rnn','gru']

        self.train_data = pd.read_csv(train_dfn, names=COLUMNS, skipinitialspace=True)
        self.test_data = pd.read_csv(test_dfn, names=COLUMNS, skipinitialspace=True, skiprows=1)

        self.train_data[self.label_column] = (self.train_data["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
        self.test_data[self.label_column] = (self.test_data["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

        network = tf.concat(1, [wide_inputs] + flat_vars, name="deep_concat")
        for k in range(len(n_nodes)):
            network = tflearn.fully_connected(network, n_nodes[k], activation="relu", name="deep_fc%d" % (k+1))
            if use_dropout:
                network = tflearn.dropout(network, 0.5, name="deep_dropout%d" % (k+1))
                
        network = tflearn.fully_connected(network, 1, activation="linear", name="deep_fc_output", bias=False)
        network = tf.reshape(network, [-1, 1])	# so that accuracy is binary_accuracy
