from __future__ import division, print_function, absolute_import

import numpy as np
import pandas as pd
import tensorflow as tf
import random
import types
import itertools
'''
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
'''

ITER_EXPONENTIAL = [2,4,8,32,64,128,256]
ITER_FIBONACCI = [2,3,5,8,13,21,34,55,89,144]

INDICATOR_NAME_LIST = ['SMA','Stochastics','StochasticsFast','StochRSI','Swing','T3','TEMA','TickCounter','TMA','TRIX','TSF','TSI','UltimateOscillator','VMA','VOL','VOLMA','VolumeCounter','VolumeOscillator','VolumeProfile','VolumeUpDown','VolumeZones','VROC','VWMA','WilliamsR','WMA','ZigZag','ZLEMA','WoodiesCCI','WoodiesPivots','ADL','ADX','ADXR','APZ','Aroon','AroonOscillator','ATR','BarTimer','Bollinger','BOP','BuySellPressure','BuySellVolume','CandlestickPattern','CCI','ChaikinMoneyFlow','ChaikinOscillator','ChaikinVolatility','CMO','ConstantLines','CurrentDayOHL','Darvas','DEMA','DM','DMI','DMIndex','DonchianChannel','DoubleStochastics','EaseOfMovement','EMA','FisherTransform','FOSC','HMA','KAMA','KeltnerChannel','KeyReversalDown','KeyReversalUp','LinReg','LinRegIntercept','LinRegSlope','MACD','MAEnvelopes','MAMA','MFI','Momentum','NBarsDown','NBarsUp','OBV','ParabolicSAR','PFE','Pivots','PPO','PriceOscillator','RangeCounter','RegressionChannel','RIND','ROC','RSI','RSS','RVI']
INDICATOR_PARAM_LIST = [['period'],['periodD','periodK','smooth'],['periodD','periodK'],['period'],['strength'],['period','tCount','vFactor'],['period'],['countDown','showPercent'],['period'],['period','signalPeriod'],['forecast','period'],['fast','slow'],['fast','intermediate','slow'],['period','volatilityPeriod'],['null'],['period'],['countDown','showPercent'],['fast','slow'],['null'],['null'],['null'],['period','smooth'],['period'],['period'],['period'],['deviationType','deviationValue','useHighLow'],['period'],['chopIndicatorWidth','neutralBars','period','periodEma','periodLinReg','periodTurbo','sideWinderLimit0','sideWinderLimit1','sideWinderWidth'],['priorDayHlc','width'],['null'],['period'],['interval','period'],['bandPct','period'],['period'],['period'],['period'],['null'],['numStdDev','period'],['smooth'],['null'],['null'],['pattern','trendStrength'],['period'],['period'],['fast','slow'],['mAPeriod','rOCPeriod'],['period'],['line1Value','line2Value','line3Value','line4Value'],['null'],['null'],['period'],['period'],['period'],['smooth'],['period'],['period'],['smoothing','volumeDivisor'],['period'],['period'],['period'],['period'],['fast','period','slow'],['offsetMultiplier','period'],['period'],['period'],['period'],['period'],['period'],['fast','slow','smooth'],['envelopePercentage','mAType','period'],['fastLimit','slowLimit'],['period'],['period'],['barCount','barDown','lowerHigh','lowerLow'],['barCount','barUp','higherHigh','higherLow'],['null'],['acceleration','accelerationMax','accelerationStep'],['period','smooth'],['pivotRangeType','priorDayHlc','userDefinedClose','userDefinedHigh','userDefinedLow','width'],['fast','slow','smooth'],['fast','slow','smooth'],['countDown'],['period','width'],['periodQ','smooth'],['period'],['period','smooth'],['eMA1','eMA2','length'],['period']]

BLANK_LIST = ['','','','','','','','','','','','','','']

ACTIVATION_LIST = ['linear','tanh','sigmoid','softmax','softplus','softsign','relu','relu6','leaky_relu','prelu','elu'] # 11 types
LAYER_BASIC_LIST = ['fully_connected'] # Only 'fully_connected' needs 'activation'.
LAYER_CORE_LIST = ['dropout','highway']
LAYER_RECURRENT_LIST = ['lstm','bidirectional_rnn','gru']# Recurrent layers have 'dropout' inside.


OPTIMIZER_LIST = ['Optimizer','SGD','RMSProp','Adam','Momentum','AdaGrad','Ftrl','AdaDelta'] # 8 types

print('INDICATOR_PARAM_LIST[1]=',INDICATOR_PARAM_LIST[1])
slice = random.sample(INDICATOR_PARAM_LIST, 5)
slice2 = random.sample(range(0,len(INDICATOR_PARAM_LIST)),5)
print(range(0,len(INDICATOR_PARAM_LIST)))
print('slice = ',slice)
print('slice2 = ',slice2)
for i in slice2:
  print(INDICATOR_NAME_LIST[i],'-',INDICATOR_PARAM_LIST[i])
print(isinstance(INDICATOR_NAME_LIST[0], str))
print(type(INDICATOR_NAME_LIST))
print('enumerate',[i for i, x in enumerate(INDICATOR_NAME_LIST) if x == 'StochasticsFast'][0])


'''
class RandNet(object):
  def __init__(self, feature_params_iter_mode='rand',features=['SMA','RSI'],optimizers=OPTIMIZER_LIST,activations=ACTIVATION_LIST):
    print('RandNet def __init__...')
    print('feature_params_iter_mode=',feature_params_iter_mode)
    self.prompt = '???'
'''

def BuildRandNet(feature_params_iter_mode='rand',features=['SMA','RSI'],optimizers=OPTIMIZER_LIST,activations=ACTIVATION_LIST):
  print('BuildRandNet...')
  feature_combinations_list = []
  for i in range(len(features)):
    feature_combinations_list.extend([_ for _ in itertools.combinations(features, i+1)])
  print(feature_combinations_list)
  print(type(feature_combinations_list))
  print(type(feature_combinations_list[0]))# Need to tansfer 'tuple' to 'list'
  '''
  Nested Iteration:
  1.Combinations of indicators
  2.indicator_param_iters[0],indicator_param_iters[1],indicator_param_iters[2]...
  3.param_iter_lists[0],param_iter_lists[1]...
  4.n_layers
  5.layer_type_list[n_layers]
  6.n_neurons_list[n_layers]
  7.activation_list[n_layers]
  8.optimizer

  Steps:
  1.Iter features
  2.Iter feature params
  3.Load data
  4.Iter layer structure
  5.Iter n_neurons & activation
  6.Connect layers
  7.Iter optimizer
  8.Save checkpoint
  '''
  # python reflection:
  #print('feature_params_iter_mode=',feature_params_iter_mode)# Continue here:
  if feature_params_iter_mode=='specified':
    print('specified')
  if feature_params_iter_mode=='enum':
    print('enum')
  elif feature_params_iter_mode=='rand':
    print('rand')
  elif feature_params_iter_mode=='ga':
    print('ga')
#1019 Idea: full concatenatation parse
  if feature_params_iter_mode=='enum':
    if isinstance(features[0], str):
      for i in features:
        index = [ind for ind, x in enumerate(INDICATOR_NAME_LIST) if x == i][0]
        params = INDICATOR_PARAM_LIST[index]
        param_iter_lists = []
        for j in range(len(params)):
          #param_iters[j] = GetIterByParamName(params[j])
          param_iter_lists.append(GetIterByParamName(params[j]))
        #print('param_iter_lists=',param_iter_lists)
        param_iters = [[x,y,z] for x in param_iter_lists[0] for y in param_iter_lists[1] for z in param_iter_lists[2]]
        #print('param_iters=',param_iters)

def GetIterByParamName(param_name):
  if param_name == 'period':
    return ITER_FIBONACCI
  elif param_name == 'smooth':
    return ITER_FIBONACCI
  elif param_name == 'fast':
    return ITER_FIBONACCI
  elif param_name == 'slow':
    return ITER_FIBONACCI
  # ...
  else:
    return ITER_FIBONACCI    

#Testing def:

BuildRandNet(feature_params_iter_mode='enum',features=['Stochastics','RSI','MACD'])

'''
INDICATOR_PARAM_ITER_PERIOD = ITER_FIBONACCI
INDICATOR_PARAM_ITER_SMOOTH = ITER_FIBONACCI
INDICATOR_PARAM_ITER_FAST = ITER_FIBONACCI
INDICATOR_PARAM_ITER_SLOW = ITER_FIBONACCI
'''

#def RandNet(incoming,iter_mode='seq',optimizers='SGD',activations='linear')

'''
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
'''
