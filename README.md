# RH-Support-Vector-Regression
Implementation of Epsilon SVR and RH SVR on Boston Housing Dataset using CVXOPT

Bi, J. and Bennett, K.P., 2003. A geometric approach to support vector regression. Neurocomputing, 55(1-2), pp.79-108.

The QPP which epsilon-SVR solves is not intuitive. Bi and Bennet in 2003 formulated Reduced Convex Hull SVR as a more geometric and intuitive approach to solve regression using Support Vector Machines. The approach is to turn the regression into a classification problem by converting the dataset into a dataset of double size by appending y+epsilon and y-epsilon to the feature vector (epsilon is the hyperparamter). Class +1 is assigned to one set and -1 is assigned to the other set. Finally a maximum margin classifier is fitted in the kernel space by solving the dual of the QPP formulated. 

Please follow the accompanying pdf for some tests. Thanks to Prof. Jayadeva for explaination of RH-SVR. This was orginally submitted as a bonus assignment in ELL409

# Visualisations
For visualising some of the results, I converted the boston housing problem into a single dimensional regression problem by only taking the lstat feature (since it has the highest correlation with y)

![lstat linear](https://github.com/kartikeya-badola/RH-Support-Vector-Regression/blob/master/lstat%20linear.png)

RH-SVR with Linear Kernel

![lstat Poly](https://github.com/kartikeya-badola/RH-Support-Vector-Regression/blob/master/lstat%20poly3.png)

RH-SVR with polynomial kernel (degree 3)

![lstat rbf](https://github.com/kartikeya-badola/RH-Support-Vector-Regression/blob/master/lstat%20rbf.png)

RH-SVR with radial basis kernel (gamma 5)
