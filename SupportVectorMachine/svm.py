import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt




'''
#au depart, on a un jeu de donnees : N points xi dans une matrice X (vecteurs a 2 dimensions) etiquetes dans un vecteur y (-1) ou (+1)
#nombre de ligne de X : nombre de points
#nombte de colonne de X : nombre de dimension de chaque point
'''

def fit(X, y, lr = 0.001, lambd = 0.01, iter = 9000):
    nbpoints, nbdim = X.shape
    w = np.zeros(nbdim)
    b=0

    for j in range(iter):
        for i in range(nbpoints):

            if (y[i]*(np.dot(X[i],w)+b)>=1):
                w-=lr*2*lambd*w
            else:
                w-=lr*(2*lambd*w-np.dot(y[i],X[i]))
                b+=lr*y[i]
    return w,b



def predict(Xnew, w, b):
    ynew = np.zeros(np.shape(Xnew)[0])
    for i in range(len(ynew)):
        if (np.dot(w,Xnew[i])+b >= 1 ):
            ynew[i]=1
        else :
            ynew[i]=-1
    return ynew


def accuracy(traintab,testtab):
    return 100*(1-sum(abs(traintab-testtab))/(2*len(traintab)))

X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=3)
y = np.where(y==0,-1,1)

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=123)

#affichage des points
plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c = np.where(ytrain==1, 'r', 'b'))

#affichage de la droite de séparation
w, b = fit(Xtrain, ytrain)
x_line = np.linspace(min(Xtrain[:, 0]), max(Xtrain[:, 0]), 100)  
y_line = (-w[0]*x_line-b)/w[1]
y_linelim = (-w[0]*x_line-b+1)/w[1]
y_linelim2 = (-w[0]*x_line-b-1)/w[1]


plt.ylim(min(Xtrain[:, 1])-3, max(Xtrain[:, 1])+3)

plt.plot(x_line, y_line, color = 'green') 
plt.plot(x_line, y_linelim, '-k') 
plt.plot(x_line, y_linelim2, '-k') 

plt.show()

#mesure de la précision grace à la comparaison entre le testing set et la prédiction du modèle
ytest_expe = predict(Xtest, w, b)
print(f"precision de {accuracy(ytest_expe,ytest)} %")









    

