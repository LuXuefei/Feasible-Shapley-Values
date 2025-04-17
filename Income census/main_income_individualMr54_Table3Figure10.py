# -*- coding: utf-8 -*-
"""
Analysis for Individual Mr 54 in the income consus dataset
Table 3 and Figure 10 
"""


import os
path = '...\\AdultIncome' # change path here
print(path)
os.chdir(path)

# ---> LOAD MIDDLE.SPYDATA
import sys
print(sys.getrecursionlimit())
sys.setrecursionlimit(10000)

import numpy as np
import pandas as pd
import time

pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 80)


import shap as shap_util
import pickle
#from scipy.stats import multivariate_normal

from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns


import tensorflow as tf
import keras
from keras.layers import Input, Dense, Concatenate, Dropout
from keras.models import Model
from keras import regularizers

# Sklearn imports
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
#from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# DiCE imports
import dice_ml
from dice_ml.utils import helpers  # helper functions
import recourse as rs

import JointIndiShapley

#%% Load dataset
# we want to study how people can move from income<5k (y=0) to  income>5k (y=1) ; so we look at prob(y=1) and hope it can increase
# https://archive.ics.uci.edu/ml/datasets/adult
# https://interpret.ml/DiCE/notebooks/DiCE_getting_started.html
# https://github.com/interpretml/DiCE
# We use standardized data for model training, counterfactual finding and density calculation
#
import pickle
file = open('AdultIncomeDataStd.pkl', 'rb')
X_train = pickle.load(file)
y_train = pickle.load(file)
X_valid = pickle.load(file)
y_valid = pickle.load(file)
Xraw_mean = pickle.load(file)
Xraw_std = pickle.load(file)
file.close()

def build_prob_model():
    # build prob_model (ooD detector), which predicts whether x is 1  (on-mainfold) 0 (off-manifold)
    input_prob = Input(shape=(X_train.shape[1],))
    layer1_prob = Dense(200, activation="relu",kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(input_prob)
    layer2_prob = Dense(200, activation="relu",kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(layer1_prob)
    layer3_prob = Dense(200, activation="relu",kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(layer2_prob)
    out_prob = Dense(1)(layer3_prob)

    model_prob = Model(inputs=input_prob, outputs=[out_prob])
    opt = keras.optimizers.SGD(learning_rate=1e-3, momentum = 0.9)
    #model_prob.compile(optimizer=opt, loss='mean_squared_error')
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)
    model_prob.compile(optimizer=opt, loss=loss)

    model_prob.load_weights('prob_model_deep_1234.ckpt')
    return model_prob


model_prob = build_prob_model()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
 
def prob_NCE(x):     #----> calculate prob!
    h = model_prob.predict(x,verbose=0)
    G =  np.clip(sigmoid(h),0.01,0.99)
    return G / (1-G) / 100#17

def pdf(X):
    return prob_NCE(X).flatten()

# Define threshold as the 5-percentile of likelihood of the training dataset
threshold = np.percentile(pdf(X_train),5)
#%% Train a random foresest

n = len(X_train)#5000 # training size
X_train = X_train.iloc[:n,]
y_train = y_train*1
y_train = y_train[:n]
y_valid = y_valid*1

# declare parameters
params = {
            'objective':'binary:logistic',
            'max_depth': 4,
            'alpha': 10,
            'learning_rate': 1.0,
            'n_estimators':100,
            'random_state':12345
        }
# instantiate the classifier 
xgb_clf = XGBClassifier(**params)
model = xgb_clf.fit(X_train, y_train)

# Check Model perfromance
# return mean accuracy
ypred_train = model.predict(X_train)
print(classification_report(y_train, ypred_train))
# out-of-sample performance
ypred_valid = model.predict(X_valid)
print(classification_report(y_valid, ypred_valid))

#%% Find quary indices with output y=0
X_target = X_valid.loc[(ypred_valid == y_valid) & (y_valid == 0),:] 
nt = len(X_target)
X_target = X_target.iloc[:nt,:]

#%% Selec a Young person
baseidx =22
#calculation with no infeasible point, young people: 22 age 29
x0 = X_target.iloc[baseidx:baseidx+1,:]
#baseind = 20, only 4 CF found and all too small likelihood; 21 no CF found

#%% Dice algorithm

dicetraindata = X_train.copy()
dicetraindata['income'] = y_train
ddice = dice_ml.Data(dataframe=dicetraindata, 
                     continuous_features=list(X_train.columns.values),#list(X_train.columns[[0,2,8,9,10]].values), 
                     outcome_name='income')
# Using sklearn backend
mdice = dice_ml.Model(model=model, backend="sklearn")
# Using method=kdtree for generating CFs
# kdtree KD Tree Search (for counterfactuals from a given training dataset)
exp = dice_ml.Dice(ddice, mdice, method='random') #random

features_to_vary=['Age','Workclass','Education-Num','Occupation',
                  'Capital Gain','Capital Loss','Hours per week']

print('Baseline case: \n', round(x0*Xraw_std+Xraw_mean))
model.predict_proba(x0)   
#mean Capital Gain = 1077
permitted_range={'Age': [x0.iloc[0]['Age'], x0.iloc[0]['Age']+ 5/Xraw_std['Age']], #x0.iloc[0]['Age']+ 100/Xraw_std['Age']
                 'Hours per week': [(0-Xraw_mean['Hours per week'])/Xraw_std['Hours per week'], (48-Xraw_mean['Hours per week'])/Xraw_std['Hours per week']],
                 'Capital Gain': [x0.iloc[0]['Capital Gain'], max((5000-Xraw_mean['Capital Gain'])/Xraw_std['Capital Gain'], x0.iloc[0]['Capital Gain']*(1.2**5)) ],
                 'Capital Loss': [x0.iloc[0]['Capital Loss'], min((2000-Xraw_mean['Capital Loss'])/Xraw_std['Capital Loss'], x0.iloc[0]['Capital Loss']) ],
                 #'Marital Status': ([0,1,2,3,5]-Xraw_mean['Marital Status'])/Xraw_std['Marital Status'],
                 #'Education-Num': [x0.iloc[0]['Education-Num'], (16-Xraw_mean['Education-Num'])/Xraw_std['Education-Num']]                 
                 'Education-Num': [x0.iloc[0]['Education-Num'],x0.iloc[0]['Education-Num']+ 5/Xraw_std['Education-Num']]
                 } #np.inf


#%% Calculation with infeasible points
baseidx =54
#calculation involve infeasible points (no education change!):
features_to_vary=['Age','Workclass','Occupation',
                  'Capital Gain','Capital Loss','Hours per week']
#39 -- threshold 7%, 43 -- threshold 5 or 7%; 54, age 45 a bit old to change oq, edu can change
x0 = X_target.iloc[baseidx:baseidx+1,:]

print('Baseline case: \n', round(x0*Xraw_std+Xraw_mean))
try:
    #start_time = time.time()    
    ep = exp.generate_counterfactuals(x0, total_CFs=5, desired_class='opposite',
                                      #feature_weights=feature_weights, 
                                      permitted_range  = permitted_range,
                                      features_to_vary=features_to_vary,
                                      posthoc_sparsity_algorithm ='binary',
                                      random_seed=333) #333
    #time_cal= time.time()- start_time  \
    #ep.visualize_as_dataframe()
    CFdf= ep.cf_examples_list[0].final_cfs_df.drop(columns = 'income')  
    print('CF case: \n', round(CFdf*Xraw_std+Xraw_mean))
except:
    print('CF not find')  
    
print('CF cases: \n', round(CFdf*Xraw_std+Xraw_mean))
model.predict_proba(CFdf)   
print(pdf(CFdf) >= threshold)    
#%% Example of infeasible points, constrained Shapley values
pdf(CFdf) >= threshold

CFdf = CFdf.loc[pdf(CFdf)>= threshold,:]

BJIshapley = pd.DataFrame(np.zeros(CFdf.shape), columns=CFdf.columns)  
#
for baseind in range(CFdf.shape[0]):
    #baseind = 0
    x0 = x0
    x1 = CFdf.iloc[baseind:baseind+1,:]
    
    if ~x1.isnull().values.any(): # if cf not null   
        def f(X, model = model, pdf = pdf, scoreind = 1,     threshold = threshold, x0 = x0):
            return model.predict_proba(X)[:,scoreind]*(pdf(X)>=threshold)  + model.predict_proba(x0)[:,scoreind]*(pdf(X)<threshold) 
        
        start_time = time.time()
        calset,U, DXorg, yy, ff, phiorg, ffshape, phishapeid= JointIndiShapley.finitechangesInd(x0,x1,f,Torder = 10)
        time_cal= time.time()- start_time
        
        print('Instance No. ',str(baseind),'; Calculate time= ', str(time_cal))  
        print('Nr. of Infeasible points: ',str(sum(pdf(DXorg) < threshold)))
        
        BJIshapley.iloc[baseind,:] = phishapeid.flatten()
    else:
        BJIshapley.iloc[baseind,:] = np.full((len(X_target.columns),),np.nan)
#%% Example of infeasible points, Original Baseline Shapley values
BaseShapley = pd.DataFrame(np.zeros(CFdf.shape), columns=CFdf.columns)  
#
for baseind in range(CFdf.shape[0]):
    #baseind = 0
    x0 = x0
    x1 = CFdf.iloc[baseind:baseind+1,:]
    
    if ~x1.isnull().values.any(): # if cf not null   
        def f(X, model = model, pdf = pdf, scoreind = 1,     threshold = threshold, x0 = x0):
            return model.predict_proba(X)[:,scoreind]
             
        start_time = time.time()
        calset,U, DXorg, yy, ff, phiorg, ffshape, phishapeid= JointIndiShapley.finitechangesInd(x0,x1,f,Torder = 10)
        time_cal= time.time()- start_time
        
        print('Instance No. ',str(baseind),'; Calculate time= ', str(time_cal))  
        print('Nr. of Infeasible points: ',str(sum(pdf(DXorg) < threshold)))
        
        BaseShapley.iloc[baseind,:] = phishapeid.flatten()
    else:
        BaseShapley.iloc[baseind,:] = np.full((len(X_target.columns),),np.nan)

#%% Save results
# file = open('Income_ind_'+ str(baseidx)+'.pkl','wb')
# pickle.dump(BJIshapley, file)
# pickle.dump(x0, file)
# pickle.dump(CFdf, file)
# pickle.dump(BaseShapley, file)
# file.close()
baseidx = 54
# load data
file = open('Income_ind_'+ str(baseidx)+'.pkl' ,'rb')
BJIshapley = pickle.load(file)
x0 = pickle.load(file)
CFdf = pickle.load(file)
BaseShapley= pickle.load(file)
file.close()


rename_dict= {
    'Age':'X1',
    'Workclass':'X2', 
    'Education-Num':'X3',   
    'Occupation':'X4',
    'Capital Gain':'X5', 
    'Capital Loss':'X6',
    'Hours per week':'X7',    
    'Marital Status':'X8', 
    'Relationship':'X9',
    'Race':'X10', 
    'Sex':'X11',
    'Country':'X12',
    }
BJIshapley.rename(columns=rename_dict, inplace=True)
BaseShapley.rename(columns=rename_dict, inplace=True)
CFdf.rename(columns=rename_dict, inplace=True)
#%% Infeasible point case - feasible shapley
refx0 = pd.concat([x0] * CFdf.shape[0], ignore_index=True)
refx0.rename(columns=rename_dict, inplace=True)
compareind = refx0 != CFdf

dfplot = BJIshapley.copy()
yrange = [0,0.42]
# create the bar plot with error bars
fig, axes = plt.subplots(1, 1, figsize=(6,4))
sns.set_theme(style="whitegrid")
axes = sns.barplot(ax = axes, data=pd.DataFrame(dfplot.loc[0,compareind.iloc[0,]]).transpose() ,ci=95)#ci='sd')
axes.tick_params(axis='x',labelsize=20)
axes.tick_params(axis='y',labelsize=20)
#axes.set_title('Counterfactual 1',fontsize = 20)
axes.set_ylim(yrange)
fig.tight_layout()
#plt.show()
plotname = 'Income_ind'+str(baseidx)+'.pdf'
plt.savefig(plotname, format="pdf", bbox_inches="tight")

#%% Infeasible point case - Baseline shapley
refx0 = pd.concat([x0] * CFdf.shape[0], ignore_index=True)
refx0.rename(columns=rename_dict, inplace=True)
compareind = refx0 != CFdf

dfplot = BaseShapley.copy()
yrange = [0,0.42]
# create the bar plot with error bars
fig, axes = plt.subplots(1, 1, figsize=(6,4))
sns.set_theme(style="whitegrid")
axes = sns.barplot(ax = axes, data=pd.DataFrame(dfplot.loc[0,compareind.iloc[0,]]).transpose() ,ci=95)#ci='sd')
axes.tick_params(axis='x',labelsize=20)
axes.tick_params(axis='y',labelsize=20)
#axes.set_title('Counterfactual 1',fontsize = 20)
axes.set_ylim(yrange)
fig.tight_layout()
plotname = 'Income_ind'+str(baseidx)+'Base.pdf'
plt.savefig(plotname, format="pdf", bbox_inches="tight")
