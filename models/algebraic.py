import numpy as np

###################### Models GPCA ###################### 
#GPCA  #PCA if beta = 0 #LC if beta = 1
def  gpca(X,W,beta,n_components):
    X = X - np.mean(X,axis=0)
    X = np.transpose(X)
    k = n_components
    XtX = np.transpose(X) @ X
    I = np.eye(XtX.shape[0])
    L = np.diag(np.sum(W,axis=0)) - W
    lambda_n = np.max(np.linalg.eigvals(XtX))
    epsilon_n = np.max(np.linalg.eigvals(L))
    G = (1-beta)*(I - XtX/lambda_n)+beta*L/epsilon_n 
    g_eigvalues, g_eigvec = np.linalg.eig(G)
    g_eigvalues = np.real(g_eigvalues)
    g_eigvec = np.real(g_eigvec)
    increasing_order_eigvalues = np.argsort(  g_eigvalues)
    Q = g_eigvec[:,increasing_order_eigvalues[:k]]
    U = X @ Q 
    return Q, U

###################### Models GMCCA ###################### 
#GMCCA #MCCA if gamma = 0
def gmmca(X,W,gamma,n_components):
    r = 1e-4
    try:
        nview = X.shape[0]
    except:
        nview = len(X)
    X = [ np.transpose( X[k] - np.mean(X[k],axis=0) )for k in range(nview) ]
    Xt = [np.transpose(x) for x in X ]
    inv_X_Xt = [np.linalg.inv(X[k] @ Xt[k] + r*np.eye(X[k].shape[0])) for k in range(len(X))]
    L = np.diag(np.sum(W,axis=0)) - W
    C = np.sum([ Xt[k] @ inv_X_Xt[k] @ X[k]  for k in range(len(X))],axis=0) - gamma * L
    g_eigvalues, g_eigvec = np.linalg.eigh(C)
    decreasing_order_eigvalues = np.argsort( - g_eigvalues)
    St = g_eigvec[:,decreasing_order_eigvalues[:n_components]]
    U = [ inv_X_Xt[k] @ X[k] @ St   for k in range(len(X))]
    Ux = [ Xt[k] @ U[k]  for k in range(len(X))]
    return St, U, Ux