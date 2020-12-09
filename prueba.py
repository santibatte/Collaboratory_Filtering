# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 12:54:21 2020

@author: joser
"""

import pandas as pd
import numpy as np
import numpy.ma as ma


links = pd.read_csv("links_small.csv")
ratings = pd.read_csv("ratings_small.csv")



ratings = ratings.drop("timestamp",axis=1)
ratings_wide = ratings.pivot_table(index = 'userId',
                                   columns = 'movieId',
                                   values = 'rating')



#### Matrices 


links_values = np.array(links)
links_features = np.array(links.columns)


ratings_values = np.array(ratings_wide)
ratings_features = np.array(ratings_wide.columns)


#### Collaborative Filtering 

def CostFunction(Y,U,V,lambd=.02):
    
    cost = .5*np.linalg.norm(np.nan_to_num(Y-U@V),"fro")**2
    
    reg = .5*lambd*np.linalg.norm(U)+.5*lambd*np.linalg.norm(V)
    
    return cost + reg
    
    
def dCostFunction_dU(Y,U,V,lambd=.02):
    return -(np.nan_to_num(Y-U@V))@V.T+lambd*U

    
def dCostFunction_dV(Y,U,V,lambd=.02):
    return -U.T@(np.nan_to_num(Y-U@V))+lambd*V

#http://ethen8181.github.io/machine-learning/recsys/1_ALSWR.html

def CollaborativeFiltering(Y,max_iter=10,k=2,eta=.0002,lambd=.02):
    
    m,n = Y.shape
    
    U=np.random.uniform(low = 0,high = (1/np.sqrt(k)),size = [m,k] )
    V=np.random.uniform(low = 0,high = (1/np.sqrt(k)),size = [k,n] )
    
    i,j,z= 0,0,0
    while i<=max_iter:
        j,z = 0,0
        
        Uold = U
        while ((j <= max_iter) or np.linalg.norm(U-Uold)>=.001):
            Uold=U
            U = U-eta*dCostFunction_dU(Y,U,V,lambd)
            print(np.linalg.norm(U-Uold))
            j+=1
         
        Vold = V
        while  ((z <= max_iter) or np.linalg.norm(V-Vold)>=.001):
            Vold = V
            V = V-eta*dCostFunction_dV(Y,U,V,lambd)
            print(np.linalg.norm(V-Vold))
            z+=1
    
        i+=1
    return U,V



Y=ratings_values


U,V = CollaborativeFiltering(Y,max_iter=100,k=2,eta=.02,lambd=2)


###################
#### Intento 2 ####
###################





  
def Value_U(Y,U,V,lambd=.02):
    I=np.identity(U.shape[1])
    
    U_ls = np.linalg.solve(V@V.T+lambd*I,(np.nan_to_num(Y,0)@V.T).T)
    
    return U_ls.T


  
def Value_V(Y,U,V,lambd=.02):
     I=np.identity(V.shape[0])
     V_ls = np.linalg.solve(U.T@U+lambd*I,(np.nan_to_num(Y,0).T@U).T)
     return V_ls

#http://ethen8181.github.io/machine-learning/recsys/1_ALSWR.html

def CollaborativeFiltering_2(Y,max_iter=1000,k=2,lambd=10):
    
    m,n = Y.shape
    
    U=np.random.uniform(low = 0, high = (1/np.sqrt(k)), size = [m,k] )
    V=np.random.uniform(low = 0, high = (1/np.sqrt(k)), size = [k,n] )
    
    i= 0
    
    
    while i<=max_iter:
        Uold=U
        U=Value_U(Y,U,V,lambd=lambd)
        Vold=V
        V=Value_V(Y,U,V,lambd=lambd)
        #print(np.linalg.norm(U-Uold),np.linalg.norm(V-Vold))
        
        if ((np.linalg.norm(U-Uold)<=.000001) and (np.linalg.norm(V-Vold)<=.000001)):
            break
        i+=1
        
        
        
        
    return U,V



Y = ratings_values

ngrid=200

x1_grid = np.arange(1,100)
x2_grid = np.linspace(1,20)
X1, X2 = np.meshgrid(x1_grid, x2_grid)

positions = np.vstack([X1.ravel(), X2.ravel()]).T
z=list()
for i in positions:
    U,V = CollaborativeFiltering_2(Y,max_iter=1000,k=int(i[0]),lambd=i[1])
    z.append(CostFunction(Y,U,V,lambd=i[1]))
