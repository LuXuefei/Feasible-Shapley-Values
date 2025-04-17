# -*- coding: utf-8 -*-
"""
Step 0. Build Machine Learning Model + Identify counterfactuals using Ustun's recourse CFAlg
Step 3. Calcuate (Feasible) Shapley values 
"""


#%% Load libraries and dataset
import os
path = '...\\GermanCredit' # change path here
print(path)
os.chdir(path)


from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Concatenate, Dropout
from keras.models import Model
from keras import regularizers
from tensorflow.python.keras.utils.generic_utils import class_and_config_for_serialized_keras_object
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
from itertools import combinations


# DiCE imports
import dice_ml
from dice_ml.utils import helpers  # helper functions


import math
import tensorflow as tf
import keras
#import copy

from keras import backend as K
from keras.models import load_model
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import classification_report


from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns
import time

import JointIndiShapley

pd.options.display.max_columns = 30
pd.set_option('display.max_rows', 80)
#%% Load dataset
def build_dataset(random_seed):
    dataset = pd.read_csv('german_processed.csv')
    #gender = dataset['Gender']
    dataset = dataset.drop(columns = ['PurposeOfLoan','OtherLoansAtStore']) #
    GoodCustomer = {-1: 0, 1:1}
    dataset = dataset.replace({'GoodCustomer': GoodCustomer})
    Gender = {'Female': 0, 'Male':1}
    dataset = dataset.replace({'Gender': Gender})
    y_raw = dataset['GoodCustomer']
    X_raw = dataset.drop(columns = ['GoodCustomer'])
    
    Xraw_mean = X_raw.mean()
    Xraw_std = X_raw.std()

    # normalize data (this is important for model convergence)
    for k in ['Age','LoanDuration','LoanAmount']:
        X_raw[k] -= X_raw[k].mean()
        X_raw[k] /= X_raw[k].std()
    X_train, X_valid, y_train, y_valid = train_test_split(X_raw, y_raw, test_size=0.2, random_state=random_seed-3)
    #X_train_IfMale = X_train['Gender']
    #X_valid_IfMale = X_valid['Gender']
    #X_train = X_train.drop(columns = ['Gender']) #
    #X_valid = X_valid.drop(columns = ['Gender']) #
    dtypes = list(zip(X_train.dtypes.index, map(str, X_train.dtypes)))
    X_train_array = np.array([X_train[k].values for k,t in dtypes]).T
    X_valid_array = np.array([X_valid[k].values for k,t in dtypes]).T
    return X_train, X_valid, y_train, y_valid, X_train_array, X_valid_array, Xraw_mean, Xraw_std

random_seed = 4321
X_train, X_valid, y_train, y_valid, X_train_array, X_valid_array, Xraw_mean, Xraw_std = build_dataset(random_seed)

#%% load prob model

random_seed = 1234
def build_prob_model():
    # build prob_model (ooD detector), which predicts whether x is 1  (on-mainfold) 0 (off-manifold)
    input_prob = Input(shape=(X_train_array.shape[1],))
    layer1_prob = Dense(200, activation="relu",kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(input_prob)
    layer2_prob = Dense(200, activation="relu",kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(layer1_prob)
    layer3_prob = Dense(200, activation="relu",kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(layer2_prob)
    out_prob = Dense(1)(layer3_prob)

    model_prob = Model(inputs=input_prob, outputs=[out_prob])
    opt = keras.optimizers.SGD(learning_rate=1e-3, momentum = 0.9)
    #model_prob.compile(optimizer=opt, loss='mean_squared_error')
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)
    model_prob.compile(optimizer=opt, loss=loss)

    model_prob.load_weights('./german_prob_model_deep_{}.ckpt'.format(random_seed))
    return model_prob


model_prob = build_prob_model()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
 
def prob_NCE(x):     #----> calculate prob!
    h = model_prob.predict(x,verbose=None)
    G =  np.clip(sigmoid(h),0.01,0.99)
    return G / (1-G) / 100#17

def pdf(X):
    return prob_NCE(X).flatten()

#%% Train a ML, use Ustun
params = {'C': 1,
 'class_weight': 'balanced',
 'max_iter': 100,
 'penalty': 'l2',
 'solver': 'liblinear'}

# Configure and train the XGBoost classifier
log_clf = LogisticRegression(**params)
model = log_clf.fit(X_train, y_train)


from sklearn.metrics import f1_score
# Check Model perfromance
# return mean accuracy
ypred_train = model.predict(X_train)
print(classification_report(y_train, ypred_train))
# out-of-sample performance
ypred_valid = model.predict(X_valid)
print(classification_report(y_valid, ypred_valid))
# Calculate the f1 score on the test set
print('training macro f1: ', f1_score(y_train, ypred_train, average='macro') )
print('test macro f1: ', f1_score(y_valid, ypred_valid, average='macro'))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_train, ypred_train) #row for actual
confusion_matrix(y_valid, ypred_valid) #row for actua
#%% Find CFs: settings
import recourse as rs
 #['Age','LoanDuration','LoanAmount']
#(X_train-Xraw_mean)/Xraw_std

action_set = rs.ActionSet(X = X_train)
action_set['Age'].actionable = False
action_set['Single'].actionable = False
action_set['JobClassIsSkilled'].actionable = False
action_set['ForeignWorker'].actionable = False
action_set['OwnsHouse'].actionable = False
action_set['RentsHouse'].actionable = False
action_set['CriticalAccountOrLoansElsewhere'].step_direction = -1
action_set['CheckingAccountBalance_geq_0'].step_direction = 1

coefficients = model.coef_[0]
action_set.set_alignment(model)

#%% filter those y_valid ==0 in validation dataset
# take all instances predicted as 0 in the dataset
X_target = pd.concat([X_train.loc[ypred_train == 0,:] , X_valid.loc[ypred_valid == 0,:] ], axis=0)
nt = len(X_target) #114
#%% for each instance, find CF
CFdf = pd.DataFrame(np.zeros((nt, len(X_target.columns))), columns=X_target.columns)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# x0 = X_train.iloc[baseind:baseind+1,]#pd.DataFrame(X_train_array[baseind:baseind+1,:])
# y_train[baseind]
for baseind in range(nt):
    #baseind = 0
    x0 = X_target.iloc[baseind:baseind+1,:]
       
    #permitted_range={'Age': [x0.iloc[0]['Age'],np.inf]}

    # find CFs
    fs = rs.Flipset(x = x0.values, action_set = action_set, clf = model)
    fs.populate(enumeration_type = 'distinct_subsets', total_items = 1)
    CFdf.iloc[baseind,:]  = x0.values[0] + fs.actions[0]
    
#%% Calculate Feasible Shapley
# Define threshold as the 5-percentile of likelihood of the training dataset
threshold = np.percentile(pdf(X_train),5) 

X_target_notfoundCF = X_target.loc[CFdf['Age'].isnull().array,:]
CFdf_notfoundCF = CFdf.loc[CFdf['Age'].isnull(),:]

X_target_novalidCF = X_target.loc[pdf(CFdf)<threshold,:]
CFdf_novalidCF = CFdf.loc[pdf(CFdf)<threshold,:]

X_target_valid = X_target.loc[pdf(CFdf)>= threshold,:]
CFdf_valid = CFdf.loc[pdf(CFdf)>= threshold,:]

BJIshapley = pd.DataFrame(np.zeros(X_target_valid.shape), columns=X_target_valid.columns)  
InfeasibleInvol = np.zeros(X_target_valid.shape[0])
NrMutFeature = np.zeros(X_target_valid.shape[0])
CalcTime = np.zeros(X_target_valid.shape[0])

BJIshapley = pd.DataFrame(np.zeros(X_target_valid.shape), columns=X_target_valid.columns)  
for baseind in range(X_target_valid.shape[0]):
    #baseind = 0
    x0 = X_target_valid.iloc[baseind:baseind+1,:]
    x1 = CFdf_valid.iloc[baseind:baseind+1,:]
    
    if ~x1.isnull().values.any(): # if cf not null   
        def f(X, model = model, pdf = pdf, scoreind = 1,     threshold = threshold, x0 = x0):
            return model.predict_proba(X)[:,scoreind]*(pdf(X)>=threshold)  + model.predict_proba(x0)[:,scoreind]*(pdf(X)<threshold) 
        
        #616sec
        start_time = time.time()
        calset,U, DXorg, yy, ff, phiorg, ffshape, phishapeid= JointIndiShapley.finitechangesInd(x0,x1,f,Torder = 10)
        time_cal= time.time()- start_time

        print('Instance No. ',str(baseind),'; Calculate time= ', str(time_cal))  
        
        BJIshapley.iloc[baseind,:] = phishapeid.flatten()
        InfeasibleInvol[baseind] = sum(pdf(DXorg) < threshold)
        NrMutFeature[baseind] = (x0.iloc[0] != x1.iloc[0]).sum()
        CalcTime[baseind] = time_cal
    else:
        BJIshapley.iloc[baseind,:] = np.full((len(X_target.columns),),np.nan)


#%% Calculate Baseline Shapley
BaseShapley = pd.DataFrame(np.zeros(X_target_valid.shape), columns=X_target_valid.columns)  
CalcTime_Base = np.zeros(X_target_valid.shape[0])
for baseind in range(X_target_valid.shape[0]):
    #baseind = 0
    x0 = X_target_valid.iloc[baseind:baseind+1,:]
    x1 = CFdf_valid.iloc[baseind:baseind+1,:]
    
    if ~x1.isnull().values.any(): # if cf not null   
        def f(X, model = model, pdf = pdf, scoreind = 1,     threshold = threshold, x0 = x0):
            return model.predict_proba(X)[:,scoreind]
        
        #616sec
        start_time = time.time()
        calset,U, DXorg, yy, ff, phiorg, ffshape, phishapeid= JointIndiShapley.finitechangesInd(x0,x1,f,Torder = 10)
        time_cal= time.time()- start_time

        print('Instance No. ',str(baseind),'; Calculate time= ', str(time_cal))  
        
        BaseShapley.iloc[baseind,:] = phishapeid.flatten()
        CalcTime_Base[baseind] = time_cal
    else:
        BaseShapley.iloc[baseind,:] = np.full((len(X_target.columns),),np.nan)

#%% save data
BJIshapley.to_excel("GermanCredit_ustun_group.xlsx",index=False)
file = open('GermanCredit_ustun.pkl','wb')
pickle.dump(BJIshapley, file)
pickle.dump(X_target, file)
pickle.dump(CFdf, file)
pickle.dump(X_target_notfoundCF, file)
pickle.dump(CFdf_notfoundCF, file)
pickle.dump(X_target_novalidCF, file)
pickle.dump(CFdf_novalidCF, file)
pickle.dump(X_target_valid, file)
pickle.dump(CFdf_valid, file)
pickle.dump(InfeasibleInvol, file)
pickle.dump(NrMutFeature, file)
pickle.dump(CalcTime, file)
pickle.dump(BaseShapley, file) 
pickle.dump(CalcTime_Base, file)
file.close()

#%% load data
file = open('GermanCredit_ustun.pkl', 'rb')
BJIshapley = pickle.load(file)
X_target = pickle.load(file)
CFdf = pickle.load(file)
X_target_notfoundCF = pickle.load(file)
CFdf_notfoundCF = pickle.load(file)
X_target_novalidCF = pickle.load(file)
CFdf_novalidCF= pickle.load(file)
X_target_valid= pickle.load(file)
CFdf_valid= pickle.load(file)
InfeasibleInvol= pickle.load(file)
NrMutFeature= pickle.load(file)
CalcTime= pickle.load(file)
BaseShapley= pickle.load(file)
CalcTime_Base= pickle.load(file)
file.close()

#X_target_valid.loc[:,['Age','LoanDuration','LoanAmount']]= X_target_valid.loc[:,['Age','LoanDuration','LoanAmount']]*Xraw_std.loc[['Age','LoanDuration','LoanAmount']]+Xraw_mean[['Age','LoanDuration','LoanAmount']]
rename_dict= {
    'LoanDuration':'X1',
    'LoanAmount':'X2', 
    'LoanRateAsPercentOfIncome':'X3',
    'YearsAtCurrentHome':'X4',
    'NumberOfOtherLoansAtBank':'X5', 
    'NumberOfLiableIndividuals':'X6',
    'HasTelephone':'X7',
    'CheckingAccountBalance_geq_0':'X8', 
    'CheckingAccountBalance_geq_200':'X9',
    'SavingsAccountBalance_geq_100':'X10', 
    'SavingsAccountBalance_geq_500':'X11',
    'MissedPayments':'X12',
    'NoCurrentLoan':'X13', 
    'CriticalAccountOrLoansElsewhere':'X14',
    'OtherLoansAtBank':'X15', 
    'HasCoapplicant':'X16', 
    'HasGuarantor':'X17', 
    'Unemployed':'X18', 
    'YearsAtCurrentJob_lt_1':'X19',
    'YearsAtCurrentJob_geq_4':'X20',     
     'Gender':'X21',
     'ForeignWorker':'X22', 
     'Single':'X23', 
     'Age':'X24', 
     'OwnsHouse':'X25',
     'RentsHouse':'X26',
     'JobClassIsSkilled':'X27'
    }
BJIshapley.rename(columns=rename_dict, inplace=True)
BaseShapley.rename(columns=rename_dict, inplace=True)
#%% infeasible points/ nr of feature changed
sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(1, 1, figsize=(6,4))
axes = sns.histplot(InfeasibleInvol, bins= 25)
axes.set(yscale="log")
axes.set_ylabel ('Count (in log scale)', fontsize = 15)
axes.set_xlabel ('Number of infeasible points', fontsize = 15)
fig.tight_layout()
#plt.show()
plotname = 'German_NrInfeasPoints_ustun.pdf'
plt.savefig(plotname, format="pdf", bbox_inches="tight")
sum(InfeasibleInvol != 0)/len(InfeasibleInvol) # 12 has infeasible points
np.where(InfeasibleInvol != 0)[0]


fig, axes = plt.subplots(1, 1, figsize=(6,4))
axes = sns.histplot(NrMutFeature, bins = 30, binwidth=1)
#axes.set(yscale="log")
axes.set_ylabel ('Count', fontsize = 15)
axes.set_xlabel ('Number of features changed', fontsize = 15)
fig.tight_layout()
#plt.show()
plotname = 'German_NrFeatChanged_ustun.pdf'
plt.savefig(plotname, format="pdf", bbox_inches="tight")
max(NrMutFeature) 


sum(NrMutFeature <= 3) /len(NrMutFeature)
#%% bar plot - all - constrained
sns.set_theme(style="whitegrid")
yrange =  [0, 0.115]
dfplot = BJIshapley.copy()
dfplot = dfplot.abs()
#dfplot = dfplot.drop(['Gender','ForeignWorker', 'Single', 'Age', 'OwnsHouse','RentsHouse','JobClassIsSkilled' ], axis=1)
dfplot = dfplot.drop(['X21','X22', 'X23', 'X24', 'X25','X26','X27' ], axis=1)

# create the bar plot with error bars
fig, axes = plt.subplots(1, 1, figsize=(8,4))
sns.set_theme(style="whitegrid")
axes = sns.barplot(data=dfplot,ci=95)
axes.tick_params(axis='y',labelsize = 15)
axes.tick_params(axis='x', labelrotation=90,labelsize = 15)
axes.set_ylim(yrange)
axes.set_ylabel ('Mean(|Constrained Shapley|)', fontsize = 15)
fig.tight_layout()
#plt.show()
plotname = 'German_ustun_all_mean.pdf'
plt.savefig(plotname, format="pdf", bbox_inches="tight")
#%% Barplot - max - constrained
sns.set_theme(style="whitegrid")
dfplot = BJIshapley.copy()
dfplot = dfplot.abs()
dfplot = dfplot.drop(['X21','X22', 'X23', 'X24', 'X25','X26','X27' ], axis=1)
max_values  = dfplot.max()

# create the bar plot with error bars
fig, axes = plt.subplots(1, 1, figsize=(8,4))
sns.set_theme(style="whitegrid")
axes = sns.barplot(x=max_values.index, y=max_values.values)
axes.tick_params(axis='y',labelsize = 15)
axes.tick_params(axis='x', labelrotation=90,labelsize = 15)
axes.set_ylim([0,0.5])
axes.set_ylabel ('Max(|Constrained Shapley|)', fontsize = 15)
fig.tight_layout()
#plt.show()
plotname = 'German_ustun_all_max.pdf'
plt.savefig(plotname, format="pdf", bbox_inches="tight")
#%% bar plot - all - baseline
sns.set_theme(style="whitegrid")
dfplot = BaseShapley.copy()
dfplot = dfplot.abs()
dfplot = dfplot.drop(['X21','X22', 'X23', 'X24', 'X25','X26','X27' ], axis=1)

# create the bar plot with error bars
fig, axes = plt.subplots(1, 1, figsize=(8,4))
sns.set_theme(style="whitegrid")
axes = sns.barplot(data=dfplot,ci=95)
axes.tick_params(axis='y',labelsize = 15)
axes.tick_params(axis='x', labelrotation=90,labelsize = 15)
axes.set_ylim(yrange)
axes.set_ylabel ('Mean(|Baseline Shapley|)', fontsize = 15)
fig.tight_layout()
#plt.show()
plotname = 'German_ustun_all_baseline.pdf'
plt.savefig(plotname, format="pdf", bbox_inches="tight")
#%% Barplot - max - baseline
sns.set_theme(style="whitegrid")
dfplot = BaseShapley.copy()
dfplot = dfplot.abs()
dfplot = dfplot.drop(['X21','X22', 'X23', 'X24', 'X25','X26','X27' ], axis=1)
max_values  = dfplot.max()

# create the bar plot with error bars
fig, axes = plt.subplots(1, 1, figsize=(8,4))
sns.set_theme(style="whitegrid")
axes = sns.barplot(x=max_values.index, y=max_values.values)
axes.tick_params(axis='y',labelsize = 15)
axes.tick_params(axis='x', labelrotation=90,labelsize = 15)
axes.set_ylim([0,0.5])
axes.set_ylabel ('Max(|Baseline Shapley|)', fontsize = 15)
fig.tight_layout()
#plt.show()
plotname = 'German_ustun_all_max_baseline.pdf'
plt.savefig(plotname, format="pdf", bbox_inches="tight")
#%% bar plot - gender
sns.set_theme(style="whitegrid")

dfplot0 = BJIshapley.copy()
ylabel = 'Mean(|Constrained Shapley|)'
plotname1 = 'German_ustun_male.pdf'
plotname2 = 'German_ustun_female.pdf'

# dfplot0 = BaseShapley.copy()
# ylabel = 'Mean(|Baseline Shapley|)'
# plotname1 = 'German_sub_male_baseline_high2low.pdf'
# plotname2 = 'German_sub_female_baseline_high2low.pdf'

dfplot = dfplot0.abs()
dfplot = dfplot.loc[(X_target_valid['Gender']==1).array,:]
dfplot = dfplot.drop(['X21','X22', 'X23', 'X24', 'X25','X26','X27' ], axis=1)
fig, axes = plt.subplots(1,1, figsize=(8,4))
sns.barplot(data=dfplot,ci=95)#ci='sd')
#axes.set_title('Male',fontsize = 15)
axes.set_ylabel (ylabel, fontsize = 15)
axes.set_ylim(yrange)
axes.tick_params(axis='x', labelrotation=90,labelsize = 15)
fig.tight_layout()
#plt.show()
plt.savefig(plotname1, format="pdf", bbox_inches="tight")



dfplot = dfplot0.abs()
dfplot = dfplot.loc[(X_target_valid['Gender']==0).array,:]
dfplot = dfplot.drop(['X21','X22', 'X23', 'X24', 'X25','X26','X27' ], axis=1)
fig, axes = plt.subplots(1,1, figsize=(8,4))
# Plot the count of male and female
sns.barplot(data=dfplot,ci=95)#ci='sd')
#axes.set_title('Female',fontsize = 15)
axes.set_ylabel (ylabel, fontsize = 15)
axes.set_ylim(yrange)
axes.tick_params(axis='x', labelrotation=90,labelsize = 15)
fig.tight_layout()
#plt.show()
plt.savefig(plotname2, format="pdf", bbox_inches="tight")


#%% Age
(X_target_valid.loc[:,['Age']]*Xraw_std.loc[['Age']]+Xraw_mean[['Age']]).hist()

sns.set_theme(style="whitegrid")

dfplot0 = BJIshapley.copy()
ylabel = 'Mean(|Constrained Shapley|)'
plotname1 = 'German_ustun_young.pdf'
plotname2 = 'German_ustun_old.pdf'

dfplot = dfplot0.abs()
dfplot = dfplot.loc[(X_target_valid.loc[:,['Age']]*Xraw_std.loc[['Age']]+Xraw_mean[['Age']]<=30).values,:]
dfplot = dfplot.drop(['X21','X22', 'X23', 'X24', 'X25','X26','X27' ], axis=1)
# 196
fig, axes = plt.subplots(1,1, figsize=(8,4))
# Plot the count of male and female
sns.barplot(data=dfplot,ci=95)#ci='sd')
#axes.set_title('Age <= 30', fontsize = 15)
axes.set_ylabel (ylabel, fontsize = 15)
axes.set_ylim(yrange)
axes.tick_params(axis='x', labelrotation=90,labelsize = 15)
fig.tight_layout()
#plt.show()
plt.savefig(plotname1, format="pdf", bbox_inches="tight")


dfplot = dfplot0.abs()
dfplot = dfplot.loc[(X_target_valid.loc[:,['Age']]*Xraw_std.loc[['Age']]+Xraw_mean[['Age']]>30).values,:]
dfplot = dfplot.drop(['X21','X22', 'X23', 'X24', 'X25','X26','X27' ], axis=1)

fig, axes = plt.subplots(1,1, figsize=(8,4))
# Plot the count of male and female
sns.barplot(data=dfplot,ci=95)#ci='sd')
#axes.set_title('Age > 30', fontsize = 15)
axes.set_ylabel (ylabel, fontsize = 15)
axes.set_ylim(yrange)
axes.tick_params(axis='x', labelrotation=90,labelsize = 15)
fig.tight_layout()
#plt.show()
plt.savefig(plotname2, format="pdf", bbox_inches="tight")


#%% Single

dfplot0 = BJIshapley.copy()
ylabel = 'Mean(|Constrained Shapley|)'
plotname1 = 'German_ustun_single.pdf'
plotname2 = 'German_ustun_notSingle.pdf'

dfplot = dfplot0.abs()
dfplot = dfplot.loc[(X_target_valid['Single']==1).array,:]
dfplot = dfplot.drop(['X21','X22', 'X23', 'X24', 'X25','X26','X27' ], axis=1)
#174

fig, axes = plt.subplots(1,1, figsize=(8,4))
# Plot the count of male and female
sns.barplot(data=dfplot,ci=95)#ci='sd')
#axes.set_title('Single',fontsize = 15)
axes.set_ylabel (ylabel, fontsize = 15)
axes.set_ylim(yrange)
axes.tick_params(axis='x', labelrotation=90,labelsize = 15)
fig.tight_layout()
#plt.show()
plt.savefig(plotname1, format="pdf", bbox_inches="tight")

dfplot = dfplot0.abs()
dfplot = dfplot.loc[(X_target_valid['Single']==0).array,:]
dfplot = dfplot.drop(['X21','X22', 'X23', 'X24', 'X25','X26','X27' ], axis=1)# 208

fig, axes = plt.subplots(1,1, figsize=(8,4))
# Plot the count of male and female
sns.barplot(data=dfplot,ci=95)#ci='sd')
#axes.set_title('Non Single', fontsize = 15)
axes.set_ylabel (ylabel, fontsize = 15)
axes.set_ylim(yrange)
axes.tick_params(axis='x', labelrotation=90,labelsize = 15)
fig.tight_layout()
#plt.show()
plt.savefig(plotname2, format="pdf", bbox_inches="tight")

#%% RentsHouse

dfplot0 = BJIshapley.copy()
ylabel = 'Mean(|Constrained Shapley|)'
plotname1 = 'German_ustun_rent.pdf'
plotname2 = 'German_ustun_notrent.pdf'


dfplot = dfplot0.abs()
dfplot = dfplot.loc[(X_target_valid['RentsHouse']==1).array,:]
dfplot = dfplot.drop(['X21','X22', 'X23', 'X24', 'X25','X26','X27' ], axis=1)
#103

fig, axes = plt.subplots(1,1, figsize=(8,4))
# Plot the count of male and female
sns.barplot(data=dfplot,ci=95)#ci='sd')
#axes.set_title('RentsHouse', fontsize = 15)
axes.set_ylabel (ylabel, fontsize = 15)
axes.set_ylim(yrange)
axes.tick_params(axis='x', labelrotation=90,labelsize = 15)
fig.tight_layout()
#plt.show()
plt.savefig(plotname1, format="pdf", bbox_inches="tight")


dfplot = dfplot0.abs()
dfplot = dfplot.loc[(X_target_valid['RentsHouse']==0).array,:]
dfplot = dfplot.drop(['X21','X22', 'X23', 'X24', 'X25','X26','X27' ], axis=1)

fig, axes = plt.subplots(1,1, figsize=(8,4))
# Plot the count of male and female
sns.barplot(data=dfplot,ci=95)#ci='sd')
#axes.set_title('No Rent', fontsize = 15)
axes.set_ylabel (ylabel, fontsize = 15)
axes.set_ylim(yrange)
axes.tick_params(axis='x', labelrotation=90,labelsize = 15)
fig.tight_layout()
#plt.show()
plt.savefig(plotname2, format="pdf", bbox_inches="tight")


#%% Beeswarm plot
import shap as shap_util
dfplot = BJIshapley.copy()
dfplot = dfplot.drop(['X21','X22', 'X23', 'X24', 'X25','X26','X27' ], axis=1)
X_target_validc = X_target_valid.copy()
X_target_validc.rename(columns=rename_dict, inplace=True)
X_target_validc = X_target_validc.drop(['X21','X22', 'X23', 'X24', 'X25','X26','X27' ], axis=1)

dfplot = dfplot.to_numpy()
#dfplot = dfplot.abs()
shap_util.summary_plot(dfplot, X_target_validc,show=False)
fig, ax = plt.gcf(), plt.gca()
ax.set_xlabel('Constrained Shapley', fontsize = 25 )
ax.tick_params(axis='both', labelsize = 25)
plotname = 'German_ustun_beewarm.pdf'
plt.savefig(plotname, format="pdf", bbox_inches="tight")

#%% partial dependent plot
plotindex = 4
colorindex = 4
dfplot = pd.DataFrame()
dfplot[features_to_vary[plotindex]] =  X_target_valid.loc[:, features_to_vary[plotindex] ] 
dfplot['Constrined Shapley'] =  BJIshapley.loc[:, features_to_vary[plotindex] ].values
dfplot['color'] =  BJIshapley.loc[:, features_to_vary[colorindex] ].values

plt.figure(figsize=(8, 6))
#sns.set()
sns.set_theme(style="whitegrid")
cmap_reversed = plt.colormaps.get_cmap('RdBu_r')
ax = sns.scatterplot(x=features_to_vary[plotindex], y='Constrined Shapley', hue="color",
                     palette=cmap_reversed, data=dfplot)

norm = plt.Normalize(dfplot['color'].min(), dfplot['color'].max())
sm = plt.cm.ScalarMappable(cmap=cmap_reversed, norm=norm) #RdBu
sm.set_array([])

# Remove the legend and add a colorbar
ax.get_legend().remove()
ax.figure.colorbar(sm, label = features_to_vary[colorindex] )
#cbar = ax.figure.colorbar(sm)
#cbar.set_label(features_to_vary[colorindex] )  # Add a label to the color bar

#sns.histplot(dfplot[features_to_vary[plotindex]], color='gray', stat='probability',fill=True)
#plt.show()
plotname = 'German_partial_dependent_'+features_to_vary[plotindex]+'_ustun.pdf'
plt.savefig(plotname, format="pdf", bbox_inches="tight")