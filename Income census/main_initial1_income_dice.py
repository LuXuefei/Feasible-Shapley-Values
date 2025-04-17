# -*- coding: utf-8 -*-
"""
Step 0. Build Machine Learning Model + Identify counterfactuals using DICE CFAlg
Step 3. Calcuate (Feasible) Shapley values 

FIgure 21. Estimated densities
Figure 22. Nr of feature changed/Nr of infeasible points
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# DiCE imports
import dice_ml
from dice_ml.utils import helpers  # helper functions

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

#%% load probability model
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
#%% Figure 21. histogram of training densities
# Set up the figure
plt.figure(figsize=(6,4))
fig, axes = plt.subplots(1, 1, figsize=(6,4))
axes = sns.histplot(pdf(X_train), bins = 30)
axes.set_ylabel ('Frequency', fontsize = 15)
axes.set_xlabel ('Estimate Densities', fontsize = 15)
plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label='threshold $\lambda$')
plt.legend(fontsize = 15)
fig.tight_layout()
#plt.show()
plotname = 'Income_densities.pdf'
plt.savefig(plotname, format="pdf", bbox_inches="tight")

#%% Train a random foresest
n = len(X_train)#5000 # training size
X_train = X_train.iloc[:n,]
y_train = y_train*1
y_train = y_train[:n]
y_valid = y_valid*1

# # According to UCI website, XGboost performs the best in terms of both accuracy and precision
# clf = RandomForestClassifier(random_state = 1234)
# model = clf.fit(X_train, y_train)

from xgboost import XGBClassifier
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
from sklearn.metrics import f1_score
# Calculate the f1 score on the test set
print('training macro f1: ', f1_score(y_train, ypred_train, average='macro') )
print('test macro f1: ', f1_score(y_valid, ypred_valid, average='macro'))
#%% Find Counterfactuals, Dice, setting
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
#%% filter those y_valid ==0 in validation dataset
# take all instances predicted as 0 in the dataset
X_target = pd.concat([X_train.loc[ypred_train == 0,:] , X_valid.loc[ypred_valid == 0,:] ], axis=0)
nt = len(X_target) #26002

#%% for each instance, find CF
CFdf = pd.DataFrame(np.zeros((nt, len(X_target.columns))), columns=X_target.columns)


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# x0 = X_train.iloc[baseind:baseind+1,]#pd.DataFrame(X_train_array[baseind:baseind+1,:])
# y_train[baseind]
for baseind in range(nt):
    #baseind = 0
    x0 = X_target.iloc[baseind:baseind+1,:]
       
    # permitted_range={'Age': [x0.iloc[0]['Age'], 5/Xraw_std['Age']], #x0.iloc[0]['Age']+ 100/Xraw_std['Age']
    #                  'Hours per week': [(0-Xraw_mean['Hours per week'])/Xraw_std['Hours per week'], (50-Xraw_mean['Hours per week'])/Xraw_std['Hours per week']],
    #                  #'Marital Status': ([0,1,2,3,5]-Xraw_mean['Marital Status'])/Xraw_std['Marital Status'],
    #                  'Education-Num': [x0.iloc[0]['Education-Num'], 50/Xraw_std['Age']]} #np.inf
    
    permitted_range={'Age': [x0.iloc[0]['Age'], x0.iloc[0]['Age']+ 5/Xraw_std['Age']], #x0.iloc[0]['Age']+ 100/Xraw_std['Age']
                     'Hours per week': [(0-Xraw_mean['Hours per week'])/Xraw_std['Hours per week'], (48-Xraw_mean['Hours per week'])/Xraw_std['Hours per week']],
                     'Capital Gain': [x0.iloc[0]['Capital Gain'], max((5000-Xraw_mean['Capital Gain'])/Xraw_std['Capital Gain'], x0.iloc[0]['Capital Gain']*(1.2**5)) ],
                     'Capital Loss': [(0-Xraw_mean['Capital Gain'])/Xraw_std['Capital Gain'], min((2000-Xraw_mean['Capital Loss'])/Xraw_std['Capital Loss'], x0.iloc[0]['Capital Loss']) ],
                     #'Capital Loss': [x0.iloc[0]['Capital Loss'], min((2000-Xraw_mean['Capital Loss'])/Xraw_std['Capital Loss'], x0.iloc[0]['Capital Loss']) ],
                     #'Marital Status': ([0,1,2,3,5]-Xraw_mean['Marital Status'])/Xraw_std['Marital Status'],
                     #'Education-Num': [x0.iloc[0]['Education-Num'], (16-Xraw_mean['Education-Num'])/Xraw_std['Education-Num']]                 
                     'Education-Num': [x0.iloc[0]['Education-Num'],x0.iloc[0]['Education-Num']+ 5/Xraw_std['Education-Num']]
                     } #np.inf

    # find CFs
    try:
        #start_time = time.time()    
        ep = exp.generate_counterfactuals(x0, total_CFs=1, desired_class='opposite',
                                          #feature_weights=feature_weights, 
                                          permitted_range  = permitted_range,
                                          features_to_vary=features_to_vary,
                                          posthoc_sparsity_algorithm ='binary') #,  
        #time_cal= time.time()- start_time  
        #ep.visualize_as_dataframe()
        CFdf.iloc[baseind,:] = ep.cf_examples_list[0].final_cfs_df.drop(columns = 'income')  
        print('Instance No. ',str(baseind))
    except:
        CFdf.iloc[baseind,:] = np.full((X_train.shape[1],),np.nan)    
        print('No CF found')
      
#%%  for each instance, calculate shapley values btw itself and its CF
sum(pdf(CFdf) >= threshold)

X_target_notfoundCF = X_target.loc[CFdf['Age'].isnull().array,:]
CFdf_notfoundCF = CFdf.loc[CFdf['Age'].isnull(),:]

X_target_novalidCF = X_target.loc[pdf(CFdf)<threshold,:]
CFdf_novalidCF = CFdf.loc[pdf(CFdf)<threshold,:]

X_target_valid = X_target.loc[pdf(CFdf)>= threshold,:]
CFdf_valid = CFdf.loc[pdf(CFdf)>= threshold,:]

BJIshapley = pd.DataFrame(np.zeros(X_target_valid.shape), columns=X_target_valid.columns)  
InfeasibleInvol = np.zeros(X_target_valid.shape[0])
NrMutFeature = np.zeros(X_target_valid.shape[0])
#
for baseind in range(X_target_valid.shape[0]):
    #baseind = 0
    x0 = X_target_valid.iloc[baseind:baseind+1,:]
    x1 = CFdf_valid.iloc[baseind:baseind+1,:]
    
    if ~x1.isnull().values.any(): # if cf not null   
        def f(X, model = model, pdf = pdf, scoreind = 1,     threshold = threshold, x0 = x0):
            return model.predict_proba(X)[:,scoreind]*(pdf(X)>=threshold)  + model.predict_proba(x0)[:,scoreind]*(pdf(X)<threshold) 

        start_time = time.time()
        calset,U, DXorg, yy, ff, phiorg, ffshape, phishapeid= JointIndiShapley.finitechangesInd(x0,x1,f,Torder = 10)
        time_cal= time.time()- start_time
        
        print('Instance No. ',str(baseind),'; Calculate time= ', str(time_cal))  
        
        BJIshapley.iloc[baseind,:] = phishapeid.flatten()
        # if infeasible points encountered
        InfeasibleInvol[baseind] = sum(pdf(DXorg) < threshold)
        NrMutFeature[baseind] = (x0.iloc[0] != x1.iloc[0]).sum()
    else:
        BJIshapley.iloc[baseind,:] = np.full((len(X_target_valid.columns),),np.nan)

#%% Figure 22. infeasible points/ nr of feature changed
sns.set_theme(style="whitegrid")

plt.figure(figsize=(6,5))
fig, axes = plt.subplots(1, 1, figsize=(6,5))
axes = sns.histplot(InfeasibleInvol, bins= 25)
axes.set(yscale="log")
axes.set_ylabel ('Count (in log scale)', fontsize = 15)
axes.set_xlabel ('Number of infeasible points', fontsize = 15)
fig.tight_layout()
#plt.show()
plotname = 'Income_NrInfeasPoints.pdf'
plt.savefig(plotname, format="pdf", bbox_inches="tight")
sum(InfeasibleInvol != 0)/len(InfeasibleInvol) # 1287 has infeasible points


plt.figure(figsize=(6,5))
fig, axes = plt.subplots(1, 1, figsize=(6,5))
axes = sns.histplot(NrMutFeature, binwidth=1)
#axes.set(yscale="log")
axes.set_ylabel ('Count', fontsize = 15)
axes.set_xlabel ('Number of features changed', fontsize = 15)
fig.tight_layout()
#plt.show()
plotname = 'Income_NrFeatChanged.pdf'
plt.savefig(plotname, format="pdf", bbox_inches="tight")
max(NrMutFeature)
sum(NrMutFeature<=3)/len(NrMutFeature)

#%% Caculate Baseline Shapley
BaseShapley = pd.DataFrame(np.zeros(X_target_valid.shape), columns=X_target_valid.columns)
for baseind in range(X_target_valid.shape[0]):
    # baseind = 0
    x0 = X_target_valid.iloc[baseind:baseind + 1, :]
    x1 = CFdf_valid.iloc[baseind:baseind + 1, :]

    if ~x1.isnull().values.any():  # if cf not null
        def f(X, model=model, pdf=pdf, scoreind=1, threshold=threshold, x0=x0):
            return model.predict_proba(X)[:, scoreind]


        # 616sec
        start_time = time.time()
        calset, U, DXorg, yy, ff, phiorg, ffshape, phishapeid = JointIndiShapley.finitechangesInd(x0, x1, f, Torder=10)
        time_cal = time.time() - start_time

        print('Instance No. ', str(baseind), '; Calculate time= ', str(time_cal))

        BaseShapley.iloc[baseind, :] = phishapeid.flatten()
    else:
        BaseShapley.iloc[baseind, :] = np.full((len(X_target.columns),), np.nan)

#%% SHAP
import shap
explainer = shap.Explainer(model.predict, X_target_valid)
shap_values = explainer(X_target_valid)
SHAPvalues = pd.DataFrame(shap_values.values)
SHAPvalues.columns = BJIshapley.columns
#%% save data
# BJIshapley.to_excel("Income_sub_XGboost_all_240601.xlsx",index=False)
# file = open('Income_sub_XGboost_all_240601.pkl','wb')
# pickle.dump(BJIshapley, file)
# pickle.dump(X_target, file)
# pickle.dump(CFdf, file)
# pickle.dump(X_target_notfoundCF, file)
# pickle.dump(CFdf_notfoundCF, file)
# pickle.dump(X_target_novalidCF, file)
# pickle.dump(CFdf_novalidCF, file)
# pickle.dump(X_target_valid, file)
# pickle.dump(CFdf_valid, file)
# pickle.dump(InfeasibleInvol, file)
# pickle.dump(NrMutFeature, file)
# pickle.dump(BaseShapley, file)
# pickle.dump(SHAPvalues, file)
# file.close()
