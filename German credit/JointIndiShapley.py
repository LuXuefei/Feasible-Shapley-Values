# -*- coding: utf-8 -*-
"""
Codes for computing Feasible Shapley effects
"""

import pandas as pd
import numpy as np

# required for finite change
import math
from math import factorial
from itertools import combinations
from inspect import isfunction


def sumbincoeff(n): 
    sum_n_over_k=0
    for i in range(n+1):
        sum_n_over_k = math.comb(n, i) + sum_n_over_k
    return int(sum_n_over_k)

def binomial(n,k):
    binom=(factorial(n))/(factorial(k)*factorial(n-k))
    return int(binom)

def sumbincoeffcut(n,Torder): 
    sum_n_over_k=0
    for i in range(Torder+1):
        sum_n_over_k = math.comb(n, i) + sum_n_over_k    
    return int(sum_n_over_k)




# finite change decomposition g * Indicator, with dim-reduction
def finitechangesInd(x0org,x1org,f, Torder = None):
    ######################################################
    # U: design matrix
    # DX: design matrix in original space
    # yy: function evaluaiton at DX
    # ff: first-order and interactions corresponding to U
    # phi: Total-order effects
    # ffshape: shapley version ff
    # phishape: Shapley version total order effects =  Feasible Shapley
    ######################################################
    
    n_var = len(x0org.columns)
    samevind = np.where(( np.array(x0org)== np.array(x1org) ).flatten() == True)[0].tolist()
    calset = [x for x in list(range(n_var)) if x not in set(samevind)]    
    
    x0 = x0org.iloc[:,calset]
    x1 = x1org.iloc[:,calset]
    
    n = len(calset)
    
    if Torder is None:
        Torder = min(len(x0.columns),n)
    
    
    m = sumbincoeffcut(n,Torder)
    U = np.zeros((n,m))

    for i in range(n):
        for j in range(m):
            if i==j:
                U[i,j]=1

    k=n
    for l in range(2,Torder+1):
        #a=combnk(1:n,l);
        a = list(combinations(list(range(1,n+1)),l))
        a = np.array(a)
        b=math.comb(n, l)
        counter=k
        for k in range(k+1,k+b+1): #for k=k+1:k+b;      
            for z in range(1,n+1): #for z=1:n
                for w in range(1,len(a[0])+1):    #for w=1:size(a,2)
                    if z == a[k-counter-1,w-1]:  #if z==a(k-counter,w)
                        U[z-1,k-1]=1

    i = 0
    aa = x0.iloc[0,U[:,i]==1]
    bb = x1.iloc[0,U[:,i]==1]
    DX = x0.replace(aa,bb)

    for i in range(1,m):
        aa = x0.iloc[0,U[:,i]==1]
        bb = x1.iloc[0,U[:,i]==1]
        DX = pd.concat([DX,x0.replace(aa,bb)], ignore_index=True)

    DXorg = pd.concat([x0org] * m, ignore_index=True)
    DXorg.iloc[:,calset] = DX
    

    yy =  np.zeros((1,m))
    for i in range(m):
        yy[:,i] = f(DXorg.iloc[i:i+1])

    i=0;
    ff=np.zeros((1,m))

    k=0;
    for l in range(1,Torder+1):
        a = list(combinations(list(range(1,n+1)),l))
        a = np.array(a)
        b= math.comb(n, l) 
        counter=k; 
        for i in range(k+1,k+b+1):
            ff[:,i-1]=yy[:,i-1]
            for u in range(1,k+1): 
                ep=0;
                for j in range(1,n+1): 
                    ep=U[j-1,i-1]*U[j-1,u-1]+ep 
                    
                if ep == sum(U[:,u-1]):
                    ind=1
                else:
                    ind=0

                ff[:,i-1] = ff[:,i-1] - ind*ff[:,u-1] 
            
            ff[:,i-1]=ff[:,i-1]-yy[:,m-1]
        
        k=k+b

    phi=np.zeros((1,n))
    for j in range(1,n+1):
        for i in range(1,m+1):
            phi[:,j-1] = phi[:,j-1] + ff[:,i-1]*U[j-1,i-1]

    phiorg = np.zeros((1,n_var))
    phiorg[:,calset] = phi

    ffshape=np.zeros((1,m))
    for i in range(1,m+1):
        if sum(U[:,i-1]) !=0:
            ffshape[:,i-1] = ff[:,i-1]/sum(U[:,i-1]) 
            
    phishape=np.zeros((1,n))
    for j in range(1,n+1):
        for i in range(1,m+1):
            if U[j-1,i-1] !=0:
                phishape[:,j-1] = phishape[:,j-1] + ff[:,i-1]*U[j-1,i-1]/sum(U[:,i-1])
        
    phishapeorg = np.zeros((1,n_var))
    phishapeorg[:,calset] = phishape

    return calset,U, DXorg, yy, ff, phiorg, ffshape, phishapeorg
