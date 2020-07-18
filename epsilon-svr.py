import numpy as np
import pandas as pd
from numpy import linalg

import cvxopt
import cvxopt.solvers as solvers

import random

def linear_kernel(x1, x2):
    return np.dot(x1, x2)
def polynomial_kernel(x, y, p=2):
    return ((1 + np.dot(x, y)) ** p)
def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

from cvxopt import matrix

tr=0.7

df = pd.read_csv("BostonHousing.csv")

df =df.sample(frac=1)
df1=(df-df.mean())/df.std()

datalist=[]
for i,a in enumerate(df.values):
#     print(df1.values[i][:13])
    datalist.append(np.concatenate([df1.values[i][:13],[a[13]]]))

e=2

train = datalist[:int(tr*len(datalist))]
test = datalist[int(tr*len(datalist)):]

posx = []
negx = []
# posy = []
# negy = []
for i,t in enumerate(train):
    posx.append(t[:13])
#     posy.append([t[13]+e])
    negx.append(t[:13])
#     negy.append([t[13]-e])
posx=np.array(posx)
# posy=np.array(posy)
negx=np.array(negx)
# negy=np.array(negy)

train = np.array(train)
test = np.array(test)

X=np.concatenate([posx,negx])

def prepare_H(kernel,X):
    H=np.zeros((X.shape[0],X.shape[0]))
    for i,a in enumerate(X):
        for j,b in enumerate(X):
            if(i>=posx.shape[0] and j<posx.shape[0]):
                H[i][j]=-1*(kernel(a,b))
            elif(i<posx.shape[0] and j>=posx.shape[0]):
                H[i][j]=-1*(kernel(a,b))
            else:
                H[i][j]=1*(kernel(a,b))
    H=np.array(H)
    return H

P = matrix(prepare_H(linear_kernel,X))

q=matrix(np.concatenate([train[:,13],-1*train[:,13]])+e*np.ones(X.shape[0]))

G = matrix(np.concatenate([np.eye(X.shape[0],X.shape[0]),-1*np.eye(X.shape[0],X.shape[0])]))

c=1

h = matrix(np.concatenate([c*np.ones(X.shape[0]),0*np.ones(X.shape[0])]))

A= matrix(np.concatenate([1*np.ones(X.shape[0]//2),-1*np.ones(X.shape[0]//2)]))

A=A.trans()

b = matrix(np.zeros(1))

solvers.options['show_progress'] = False
solvers.options['abstol'] = 1e-10
solvers.options['reltol'] = 1e-10
solvers.options['feastol'] = 1e-10
solvers.options['maxiters'] = 400

sol = solvers.qp(P, q, G, h, A, b)
alphas = np.array(sol['x'])

alphaplus = alphas[:len(alphas)//2]
alphaminus = alphas[len(alphas)//2:]

def getb(alphaplus,aplhaminus,train,x,y,kernel):
    s=0
    for i,a in enumerate(alphaminus):
        s=s+(a[0]-alphaplus[i][0])*kernel(train[i][:13],x)
    s=s+b
    return y-s #epsilon still remains though

for i,t in enumerate(train):
    if(i<len(alphaplus)):
        if(alphaplus[i]<c-0.1 and alphaplus[i]>0.1):
            break
    else:
        if(alphaminus[i-len(alphaplus)]<c-0.1 and alphaminus[i-len(alphaplus)]>0.1):
            break

b=getb(alphaplus,alphaminus,train,train[i][:13],train[i][13],linear_kernel)[0][0]
if(i<len(alphaplus)):
    b=b+e
else:
    b=b-e

def predict(alphaplus,aplhaminus,train,x,kernel,b):
    s=0
    for i,a in enumerate(alphaminus):
        s=s+(a[0]-alphaplus[i][0])*kernel(train[i][:13],x)
    s=s+b
    return s

mse=0
for i,d in enumerate(test):
    print(predict(alphaplus,alphaminus,train,d[:13],linear_kernel,b),d[13])
    mse=mse+((predict(alphaplus,alphaminus,train,d[:13],linear_kernel,b)-d[13]))**2
print('MSE='+str(mse/len(test)))


