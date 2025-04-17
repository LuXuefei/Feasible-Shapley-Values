# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 16:41:33 2022

@author: xuefei.lu
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
        #sum_n_over_k=(factorial(n))/(factorial(i)*factorial(n-i))+sum_n_over_k       
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
    # calset: changed features
    # U: design matrix
    # DX: design matrix in original space
    # yy: function evaluaiton at DX
    # ff: first-order and interactions corresponding to U
    # phi: Total-order effects
    # ffshape: shapley version ff
    # phishape: Shapley version total order effects =  Baseline Shapley
    ######################################################
    
    n_var = len(x0org.columns)
    samevind = np.where(( np.array(x0org)== np.array(x1org) ).flatten() == True)[0].tolist()
    calset = [x for x in list(range(n_var)) if x not in set(samevind)]    
    
    x0 = x0org.iloc[:,calset]
    x1 = x1org.iloc[:,calset]
    
    n = len(calset)
    
    if Torder is None:
        Torder = min(len(x0.columns),n)
    
    
    m = sumbincoeffcut(n,Torder) #sumbincoeff(n)
    U = np.zeros((n,m))

    for i in range(n):
        for j in range(m):
            if i==j:
                U[i,j]=1

# dim 26 - 2.6hr / 17902 out of 67108864 order <=4
    #start_time = time.time()  
    k=n
    #for l in range(2,n+1): # l = 2:n
    for l in range(2,Torder+1):
        #a=combnk(1:n,l);
        a = list(combinations(list(range(1,n+1)),l))
        a = np.array(a)
        b=math.comb(n, l)#binomial(n,l)
        counter=k
        for k in range(k+1,k+b+1): #for k=k+1:k+b;      
            for z in range(1,n+1): #for z=1:n
                for w in range(1,len(a[0])+1):    #for w=1:size(a,2)
                    if z == a[k-counter-1,w-1]:  #if z==a(k-counter,w)
                        U[z-1,k-1]=1
    
    #time_cal= time.time()- start_time  

    #U
    #col_sums = np.sum(U, axis=0)
    #U = U[:, col_sums <= Torder]

    i = 0
    aa = x0.iloc[0,U[:,i]==1]
    bb = x1.iloc[0,U[:,i]==1]
    DX = x0.replace(aa,bb)

    for i in range(1,m):
        aa = x0.iloc[0,U[:,i]==1]
        bb = x1.iloc[0,U[:,i]==1]
        DX = pd.concat([DX,x0.replace(aa,bb)], ignore_index=True)

    #DX

    DXorg = pd.concat([x0org] * m, ignore_index=True)
    DXorg.iloc[:,calset] = DX
    

    yy =  np.zeros((1,m))
    for i in range(m):
        yy[:,i] = f(DXorg.iloc[i:i+1])

    #yy
    i=0;
    ff=np.zeros((1,m))
    #count=0;
    k=0;
    for l in range(1,Torder+1):#range(1,n+1): #l=1:n
        a = list(combinations(list(range(1,n+1)),l))
        a = np.array(a)
        b= math.comb(n, l) #binomial(n,l)
        counter=k; 
        for i in range(k+1,k+b+1):#i=k+1:k+b;
            ff[:,i-1]=yy[:,i-1] # ff(i)=y(i)
            for u in range(1,k+1): #u=1:k
                ep=0;
                for j in range(1,n+1): #j=1:n
                    ep=U[j-1,i-1]*U[j-1,u-1]+ep #ep=U(j,i)*U(j,u)+ep;
                    
                if ep == sum(U[:,u-1]): #ep==sum(U(:,u))
                    ind=1
                else:
                    ind=0

                ff[:,i-1] = ff[:,i-1] - ind*ff[:,u-1] #ff(i)=ff(i)-ind*ff(u);
            
            ff[:,i-1]=ff[:,i-1]-yy[:,m-1]
        
        k=k+b

    #ff


    phi=np.zeros((1,n))
    for j in range(1,n+1):#j=1:n
        for i in range(1,m+1):#i=1:m
            phi[:,j-1] = phi[:,j-1] + ff[:,i-1]*U[j-1,i-1]#phi(j)=phi(j)+ff(i)*U(j,i);

    #phi
    phiorg = np.zeros((1,n_var))
    phiorg[:,calset] = phi

    ffshape=np.zeros((1,m))
    for i in range(1,m+1): #i=1:m
        if sum(U[:,i-1]) !=0:
            ffshape[:,i-1] = ff[:,i-1]/sum(U[:,i-1]) #ffshape(i)=ff(i)/sum(U(:,i));
            
    phishape=np.zeros((1,n))
    for j in range(1,n+1):#j=1:n
        for i in range(1,m+1):#i=1:m
            if U[j-1,i-1] !=0: #if U(j,i) ~=0
                phishape[:,j-1] = phishape[:,j-1] + ff[:,i-1]*U[j-1,i-1]/sum(U[:,i-1])#phishape(j)=phishape(j)+ff(i)*U(j,i)/sum(U(:,i));
        
    phishapeorg = np.zeros((1,n_var))
    phishapeorg[:,calset] = phishape

    return calset,U, DXorg, yy, ff, phiorg, ffshape, phishapeorg



# # finite change decomposition g * Indicator
# def finitechangesInd(x0,x1,f, Torder = None):
#     ######################################################
#     # U: design matrix
#     # DX: design matrix in original space
#     # yy: function evaluaiton at DX
#     # ff: first-order and interactions corresponding to U
#     # phi: Total-order effects
#     # ffshape: shapley version ff
#     # phishape: Shapley version total order effects =  Baseline Shapley
#     ######################################################
#     if Torder is None:
#         Torder = len(x0.columns)
    
#     n = len(x0.columns)
#     m = sumbincoeffcut(n,Torder) #sumbincoeff(n)
#     U = np.zeros((n,m))

#     for i in range(n):
#         for j in range(m):
#             if i==j:
#                 U[i,j]=1

# # dim 26 - 2.6hr / 17902 out of 67108864 order <=4
#     #start_time = time.time()  
#     k=n
#     #for l in range(2,n+1): # l = 2:n
#     for l in range(2,Torder+1):
#         #a=combnk(1:n,l);
#         a = list(combinations(list(range(1,n+1)),l))
#         a = np.array(a)
#         b=math.comb(n, l)#binomial(n,l)
#         counter=k
#         for k in range(k+1,k+b+1): #for k=k+1:k+b;      
#             for z in range(1,n+1): #for z=1:n
#                 for w in range(1,len(a[0])+1):    #for w=1:size(a,2)
#                     if z == a[k-counter-1,w-1]:  #if z==a(k-counter,w)
#                         U[z-1,k-1]=1
    
#     #time_cal= time.time()- start_time  

#     #U
#     #col_sums = np.sum(U, axis=0)
#     #U = U[:, col_sums <= Torder]

#     i = 0
#     aa = x0.iloc[0,U[:,i]==1]
#     bb = x1.iloc[0,U[:,i]==1]
#     DX = x0.replace(aa,bb)

#     for i in range(1,m):
#         aa = x0.iloc[0,U[:,i]==1]
#         bb = x1.iloc[0,U[:,i]==1]
#         DX = pd.concat([DX,x0.replace(aa,bb)], ignore_index=True)

#     #DX


#     yy =  np.zeros((1,m))
#     for i in range(m):
#         yy[:,i] = f(DX.iloc[i:i+1])

#     #yy
#     i=0;
#     ff=np.zeros((1,m))
#     #count=0;
#     k=0;
#     for l in range(1,Torder+1):#range(1,n+1): #l=1:n
#         a = list(combinations(list(range(1,n+1)),l))
#         a = np.array(a)
#         b= math.comb(n, l) #binomial(n,l)
#         counter=k; 
#         for i in range(k+1,k+b+1):#i=k+1:k+b;
#             ff[:,i-1]=yy[:,i-1] # ff(i)=y(i)
#             for u in range(1,k+1): #u=1:k
#                 ep=0;
#                 for j in range(1,n+1): #j=1:n
#                     ep=U[j-1,i-1]*U[j-1,u-1]+ep #ep=U(j,i)*U(j,u)+ep;
                    
#                 if ep == sum(U[:,u-1]): #ep==sum(U(:,u))
#                     ind=1
#                 else:
#                     ind=0

#                 ff[:,i-1] = ff[:,i-1] - ind*ff[:,u-1] #ff(i)=ff(i)-ind*ff(u);
            
#             ff[:,i-1]=ff[:,i-1]-yy[:,m-1]
        
#         k=k+b

#     #ff


#     phi=np.zeros((1,n))
#     for j in range(1,n+1):#j=1:n
#         for i in range(1,m+1):#i=1:m
#             phi[:,j-1] = phi[:,j-1] + ff[:,i-1]*U[j-1,i-1]#phi(j)=phi(j)+ff(i)*U(j,i);

#     #phi

#     ffshape=np.zeros((1,m))
#     for i in range(1,m+1): #i=1:m
#         if sum(U[:,i-1]) !=0:
#             ffshape[:,i-1] = ff[:,i-1]/sum(U[:,i-1]) #ffshape(i)=ff(i)/sum(U(:,i));
            
#     phishape=np.zeros((1,n))
#     for j in range(1,n+1):#j=1:n
#         for i in range(1,m+1):#i=1:m
#             if U[j-1,i-1] !=0: #if U(j,i) ~=0
#                 phishape[:,j-1] = phishape[:,j-1] + ff[:,i-1]*U[j-1,i-1]/sum(U[:,i-1])#phishape(j)=phishape(j)+ff(i)*U(j,i)/sum(U(:,i));
        

#     return U, DX, yy, ff, phi, ffshape, phishape





# # finite change decomposition g * Indicator full decomposition -- no dim reducnmtion
# def finitechangesIndfull(x0,x1,f):
#     ######################################################
#     # U: design matrix
#     # DX: design matrix in original space
#     # yy: function evaluaiton at DX
#     # ff: first-order and interactions corresponding to U
#     # phi: Total-order effects
#     # ffshape: shapley version ff
#     # phishape: Shapley version total order effects =  Baseline Shapley
#     ######################################################
#     n = len(x0.columns)
#     m = sumbincoeff(n)
#     U = np.zeros((n,m))

#     for i in range(n):
#         for j in range(m):
#             if i==j:
#                 U[i,j]=1

#     k=n
#     for l in range(2,n+1): # l = 2:n
#         #a=combnk(1:n,l);
#         a = list(combinations(list(range(1,n+1)),l))
#         a = np.array(a)
#         b=binomial(n,l)
#         counter=k
#         for k in range(k+1,k+b+1): #for k=k+1:k+b;
#             if k==m-1: #k==m-1
#                 U[:,k-1]=1
#             elif k==m: #k == m
#                 U[:,k-1]=0
#             else:       
#                 for z in range(1,n+1): #for z=1:n
#                     for w in range(1,len(a[0])+1):    #for w=1:size(a,2)
#                         if z == a[k-counter-1,w-1]:  #if z==a(k-counter,w)
#                             U[z-1,k-1]=1

#     #U

#     i = 0
#     aa = x0.iloc[0,U[:,i]==1]
#     bb = x1.iloc[0,U[:,i]==1]
#     DX = x0.replace(aa,bb)

#     for i in range(1,m):
#         aa = x0.iloc[0,U[:,i]==1]
#         bb = x1.iloc[0,U[:,i]==1]
#         DX = pd.concat([DX,x0.replace(aa,bb)], ignore_index=True)

#     #DX


#     yy =  np.zeros((1,m))
#     for i in range(m):
#         yy[:,i] = f(DX.iloc[i:i+1])
#     # if isfunction(clf):
#     #     for i in range(m):            
#     #         yy[:,i] = clf(DX.iloc[i:i+1])*(pdf(DX.iloc[i:i+1])>=threshold) + clf(x0)*(pdf(DX.iloc[i:i+1])<threshold) 
#     # else:
#     #     for i in range(m):
#     #         yy[:,i] = clf.predict_proba(DX.iloc[i:i+1])[:,scoreind]*(pdf(DX.iloc[i:i+1])>=threshold) + clf.predict_proba(x0)[:,scoreind]*(pdf(DX.iloc[i:i+1])<threshold) 

#     #yy
#     i=0;
#     ff=np.zeros((1,m))
#     #count=0;
#     k=0;
#     for l in range(1,n+1): #l=1:n
#         #a=combnk(1:n,l);
#         a = list(combinations(list(range(1,n+1)),l))
#         a = np.array(a)
#         b=binomial(n,l)
#         counter=k; 
#         for i in range(k+1,k+b+1):#i=k+1:k+b;
#             ff[:,i-1]=yy[:,i-1] # ff(i)=y(i)
#             for u in range(1,k+1): #u=1:k
#                 ep=0;
#                 for j in range(1,n+1): #j=1:n
#                     ep=U[j-1,i-1]*U[j-1,u-1]+ep #ep=U(j,i)*U(j,u)+ep;
                    
#                 if ep == sum(U[:,u-1]): #ep==sum(U(:,u))
#                     ind=1
#                 else:
#                     ind=0

#                 ff[:,i-1] = ff[:,i-1] - ind*ff[:,u-1] #ff(i)=ff(i)-ind*ff(u);
            
#             ff[:,i-1]=ff[:,i-1]-yy[:,m-1]
        
#         k=k+b

#     #ff


#     phi=np.zeros((1,n))
#     for j in range(1,n+1):#j=1:n
#         for i in range(1,m+1):#i=1:m
#             phi[:,j-1] = phi[:,j-1] + ff[:,i-1]*U[j-1,i-1]#phi(j)=phi(j)+ff(i)*U(j,i);

#     #phi

#     ffshape=np.zeros((1,m))
#     for i in range(1,m+1): #i=1:m
#         if sum(U[:,i-1]) !=0:
#             ffshape[:,i-1] = ff[:,i-1]/sum(U[:,i-1]) #ffshape(i)=ff(i)/sum(U(:,i));
            
#     phishape=np.zeros((1,n))
#     for j in range(1,n+1):#j=1:n
#         for i in range(1,m+1):#i=1:m
#             if U[j-1,i-1] !=0: #if U(j,i) ~=0
#                 phishape[:,j-1] = phishape[:,j-1] + ff[:,i-1]*U[j-1,i-1]/sum(U[:,i-1])#phishape(j)=phishape(j)+ff(i)*U(j,i)/sum(U(:,i));
        

#     return U, DX, yy, ff, phi, ffshape, phishape




# # finite change decomposition g * density
# def finitechangesdenfull(x0,x1,clf,pdf,scoreind=0):
#     ######################################################
#     # U: design matrix
#     # DX: design matrix in original space
#     # yy: function evaluaiton at DX
#     # ff: first-order and interactions corresponding to U
#     # phi: Total-order effects
#     # ffshape: shapley version ff
#     # phishape: Shapley version total order effects =  Baseline Shapley
#     ######################################################
#     n = len(x0.columns)
#     m = sumbincoeff(n)
#     U = np.zeros((n,m))

#     for i in range(n):
#         for j in range(m):
#             if i==j:
#                 U[i,j]=1

#     k=n
#     for l in range(2,n+1): # l = 2:n
#         #a=combnk(1:n,l);
#         a = list(combinations(list(range(1,n+1)),l))
#         a = np.array(a)
#         b=binomial(n,l)
#         counter=k
#         for k in range(k+1,k+b+1): #for k=k+1:k+b;
#             if k==m-1: #k==m-1
#                 U[:,k-1]=1
#             elif k==m: #k == m
#                 U[:,k-1]=0
#             else:       
#                 for z in range(1,n+1): #for z=1:n
#                     for w in range(1,len(a[0])+1):    #for w=1:size(a,2)
#                         if z == a[k-counter-1,w-1]:  #if z==a(k-counter,w)
#                             U[z-1,k-1]=1

#     #U

#     i = 0
#     aa = x0.iloc[0,U[:,i]==1]
#     bb = x1.iloc[0,U[:,i]==1]
#     DX = x0.replace(aa,bb)

#     for i in range(1,m):
#         aa = x0.iloc[0,U[:,i]==1]
#         bb = x1.iloc[0,U[:,i]==1]
#         DX = pd.concat([DX,x0.replace(aa,bb)], ignore_index=True)

#     #DX


#     yy =  np.zeros((1,m))
#     if isfunction(clf):
#         for i in range(m):
#             yy[:,i] = clf(DX.iloc[i:i+1])*pdf(DX.iloc[i:i+1])
#     else:
#         for i in range(m):
#             yy[:,i] = clf.predict_proba(DX.iloc[i:i+1])[:,scoreind] * pdf(DX.iloc[i:i+1])

#     #yy
            

#     #
#     i=0;
#     ff=np.zeros((1,m))
#     #count=0;
#     k=0;
#     for l in range(1,n+1): #l=1:n
#         #a=combnk(1:n,l);
#         a = list(combinations(list(range(1,n+1)),l))
#         a = np.array(a)
#         b=binomial(n,l)
#         counter=k; 
#         for i in range(k+1,k+b+1):#i=k+1:k+b;
#             ff[:,i-1]=yy[:,i-1] # ff(i)=y(i)
#             for u in range(1,k+1): #u=1:k
#                 ep=0;
#                 for j in range(1,n+1): #j=1:n
#                     ep=U[j-1,i-1]*U[j-1,u-1]+ep #ep=U(j,i)*U(j,u)+ep;
                    
#                 if ep == sum(U[:,u-1]): #ep==sum(U(:,u))
#                     ind=1
#                 else:
#                     ind=0

#                 ff[:,i-1] = ff[:,i-1] - ind*ff[:,u-1] #ff(i)=ff(i)-ind*ff(u);
            
#             ff[:,i-1]=ff[:,i-1]-yy[:,m-1]
        
#         k=k+b

#     #ff


#     phi=np.zeros((1,n))
#     for j in range(1,n+1):#j=1:n
#         for i in range(1,m+1):#i=1:m
#             phi[:,j-1] = phi[:,j-1] + ff[:,i-1]*U[j-1,i-1]#phi(j)=phi(j)+ff(i)*U(j,i);

#     #phi

#     ffshape=np.zeros((1,m))
#     for i in range(1,m+1): #i=1:m
#         if sum(U[:,i-1]) !=0:
#             ffshape[:,i-1] = ff[:,i-1]/sum(U[:,i-1]) #ffshape(i)=ff(i)/sum(U(:,i));
            
#     phishape=np.zeros((1,n))
#     for j in range(1,n+1):#j=1:n
#         for i in range(1,m+1):#i=1:m
#             if U[j-1,i-1] !=0: #if U(j,i) ~=0
#                 phishape[:,j-1] = phishape[:,j-1] + ff[:,i-1]*U[j-1,i-1]/sum(U[:,i-1])#phishape(j)=phishape(j)+ff(i)*U(j,i)/sum(U(:,i));
        

#     return U, DX, yy, ff, phi, ffshape, phishape







#%%
# # local and global importance -  with counterfactuals
# def Localfinitechanges_importance(x0,CFset,clf, desired_class):   
#     n_CFset = CFset.shape[0]
    
#     phiLocal = np.zeros((n_CFset,x0.shape[1]))
#     phishapeLocal = np.zeros((n_CFset,x0.shape[1]))
    
#     for desiredind in range(1,n_CFset+1):
#         #desiredind = 3 # n_CFset
#         x1 = CFset[desiredind-1:desiredind]
#         # Define desired class prob column
#         scoreind = np.where(clf.classes_ == desired_class)[0]

#         U, DX, yy, ff, phi, ffshape, phishape = finitechangesfull(x0,x1,clf,scoreind)

#         phiLocal[desiredind-1,:] = phi
#         phishapeLocal[desiredind-1,:] = phishape
        
        
#     phiLocalAgr = np.mean(abs(phiLocal),axis = 0)
#     phishapeLocalAgr = np.mean(abs(phishapeLocal),axis = 0)
    
#     return phiLocalAgr, phishapeLocalAgr, phiLocal, phishapeLocal



# def Globalfinitechanges_importance(query_instances, n_CFset, clf, exp, desired_class):
#     phiGlobal = np.zeros(query_instances.shape)
#     phishapeGlobal = np.zeros(query_instances.shape)
    
    
#     for ii in range(1,query_instances.shape[0]+1):
#         x0 = query_instances[ii-1:ii]
        
#         e1 = exp.generate_counterfactuals(x0, total_CFs=n_CFset, desired_class=desired_class,  posthoc_sparsity_algorithm ='binary')
#         #e1.visualize_as_dataframe()
#         CFset = e1.cf_examples_list[0].final_cfs_df
        
#         phiLocalAgr, phishapeLocalAgr,__ ,__ = Localfinitechanges_importance(x0,CFset,clf, desired_class)
        
#         phiGlobal[ii-1,:] = phiLocalAgr
#         phishapeGlobal[ii-1,:] = phishapeLocalAgr
        
#     phiLocalAgr = np.mean(abs(phiGlobal),axis = 0)
#     phishapeLocalAgr = np.mean(abs(phishapeGlobal),axis = 0)
        
#     return phiLocalAgr, phishapeLocalAgr, phiGlobal, phishapeGlobal