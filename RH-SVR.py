
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
posy = []
negy = []
for i,t in enumerate(train):
    posx.append(t[:13])
    posy.append([t[13]+e])
    negx.append(t[:13])
    negy.append([t[13]-e])
posx=np.array(posx)
posy=np.array(posy)
negx=np.array(negx)
negy=np.array(negy)

X = np.concatenate([posx,negx])

Y = np.concatenate([posy,-1*negy])

def prepare_H(kernel,X,Y):
    H=np.zeros((X.shape[0],X.shape[0]))
    for i,a in enumerate(X):
        for j,b in enumerate(X):
            if(i>=posx.shape[0] and j<posx.shape[0]):
                H[i][j]=-1*kernel(a,b)
            elif(i<posx.shape[0] and j>=posx.shape[0]):
                H[i][j]=-1*kernel(a,b)
            else:
                H[i][j]=1*kernel(a,b)
    H=np.array(H)
    H=H+ Y.dot(Y.transpose())
    return H

P = matrix(prepare_H(linear_kernel,X,Y))

q=matrix(-np.ones(X.shape[0]))

G = matrix(np.concatenate([np.eye(X.shape[0],X.shape[0]),-1*np.eye(X.shape[0],X.shape[0])]))

c=1

h = matrix(np.concatenate([c*np.ones(X.shape[0]),0*np.ones(X.shape[0])]))

A= matrix(np.concatenate([np.ones(X.shape[0]//2),-1*np.ones(X.shape[0]//2)]))

A=A.trans()

b = matrix(np.zeros(1))

solvers.options['show_progress'] = False
solvers.options['abstol'] = 1e-10
solvers.options['reltol'] = 1e-10
solvers.options['feastol'] = 1e-10
solvers.options['maxiters'] = 400

sol = solvers.qp(P, q, G, h, A, b)
alphas = np.array(sol['x'])

def getb(alphas,x,y,kernel):
    s=0
    for i,px in enumerate(posx):
        s=s+alphas[i]*(kernel(px,x)+posy[i].dot(y.transpose()))
    for i,nx in enumerate(negx):
        s=s-alphas[posx.shape[0]+i]*(kernel(nx,x)+negy[i].dot(y.transpose()))
    return 1-1*s[0]

for i in range(posx.shape[0]):
    if(alphas[i]>0.1 and alphas[i]<c-0.1):
        break
b = getb(alphas,X[i],Y[i],linear_kernel)
print(b)
def get_eta(alphas):
    eta=0
    for i,py in enumerate(posy):
        eta = eta+alphas[i]*py
    for i,ny in enumerate(negy):
        eta = eta-alphas[posy.shape[0]+i]*ny
    return eta[0]

eta=get_eta(alphas)

def predict(alphas,X,x,b,eta,kernel):
    s=0
    for i,px in enumerate(posx):
        s=s+alphas[i]*(kernel(px,x))
    for i,nx in enumerate(negx):
        s=s-alphas[posx.shape[0]+i]*(kernel(nx,x))
        
    return -1*(s[0]+b)/eta

mse=0
for i,d in enumerate(test):
    print(predict(alphas,X,d[:13],b,eta,linear_kernel),d[13])
    mse=mse+((predict(alphas,X,d[:13],b,eta,linear_kernel)-d[13]))**2
print('MSE='+str(mse/len(test)))