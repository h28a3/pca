import numpy as np
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
wine = fetch_ucirepo(id=109) 

X = wine.data.features.to_numpy()
y = wine.data.target

N,M = X.shape

mu = np.mean(X, axis=0)
cov = np.cov(X.T)
eig_val,P = np.linalg.eig(cov)

#固有値を大きい順に並び替え
eigen_id = np.argsort(eig_val)[::-1]
eig_val = eig_val[eigen_id]
P = P[:,eigen_id]

#白色化
wh=[]
for i in range(M):
    tmp=[]
    for j in range(M):
        if i==j:
            tmp.append(np.sqrt(eig_val[i]))
        else:
            tmp.append(0)
    wh.append(tmp)
wh = np.linalg.inv(wh)

#中心化
for i in range(N):
    for j in range(M):
        X[i][j] -= mu[j]

data_pca = np.dot(wh,P.T)
data_pca = np.dot(data_pca,X.T)
