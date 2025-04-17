# -*- coding: utf-8 -*-
"""
 Figure 11  Barplots for absolute feasible Shapley effects Transition from higher- to lower- income class
"""

import os
path = '...\\AdultIncome'
print(path)
os.chdir(path)

import sys
#print(sys.getrecursionlimit())
sys.setrecursionlimit(10000)

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

import shap as shap_util
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 80)

#%%
file = open('Income_sub_XGboost_all_high2low.pkl', 'rb')
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
BaseShapley= pickle.load(file)
SHAPvalues= pickle.load(file)
file.close()

features_to_vary=['Age','Workclass','Education-Num','Occupation',
                  'Capital Gain','Capital Loss','Hours per week']

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
SHAPvalues.rename(columns=rename_dict, inplace=True)

#%% Figure 11. Absolute Feasible Shapley values: High to Low
yrange =  [0, 0.3]
ylabel = r'Mean |$S_i(\widetilde{\varphi})$|'
sns.set_theme(style="whitegrid")
dfplot = BJIshapley.copy()
dfplot = dfplot.abs()
dfplot = dfplot.drop(['X8','X9', 'X10', 'X11', 'X12'], axis=1)
# create the bar plot with error bars
fig, axes = plt.subplots(1, 1, figsize=(6,4))
sns.set_theme(style="whitegrid")
axes = sns.barplot(data=dfplot,ci=95)
axes.tick_params(axis='y',labelsize = 15)
axes.tick_params(axis='x',labelsize = 15)
axes.set_ylim(yrange)
axes.set_ylabel (ylabel, fontsize = 15)
fig.tight_layout()
#plt.show()
plotname = 'Income_sub_all_high2low.pdf'
plt.savefig(plotname, format="pdf", bbox_inches="tight")

#%% Nr of transitions involving infeasible points

print("There are "+ str(sum(InfeasibleInvol))+"out of"+ str(CFdf_valid.shape[0]) +"transitions involving infeasible points.")
sum(InfeasibleInvol)/CFdf_valid.shape[0]
