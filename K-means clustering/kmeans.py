
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from random import *
from mpl_toolkits.mplot3d import Axes3D


class KMeans():

    def __init__(self, features, K, max_iter):
        self.features = features
        self.K = K
        self.max_iter = max_iter
        self.centroids = []
        self.clusters = [[] for _ in range(self.K)] 
        print(self.features, self.K, self.max_iter)
        
        self.X = datasets.make_blobs(n_samples=500, n_features=self.features, centers=self.K, cluster_std=1.05, random_state=40)

        for i in range(self.K):
            #chaque centroid est initialisé sur une position aléatoire entre la valeur min et max du jeu de donnée sur sa dimension
            centroid = []
            for j in range(self.features):
                centroid.append(uniform(min(self.X[:, j]),max(self.X[:, j])))
            self.centroids.append(centroid)
        
        self.centroids = np.array(self.centroids)

        #affichage des points si les données sont en 2d ou 3d
        if (self.features == 2) :
            plt.scatter(self.X[:, 0], self.X[:, 1], label = "data")
            plt.scatter(self.centroids[:, 0],self.centroids[:, 1], c='red', label = 'centroids')
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title("before training phase")
            plt.legend()
            plt.show()
        elif (self.features == 3) : 
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(self.X[:, 0], self.X[:, 1], self.X[:, 2], label = 'data')
            ax.scatter(self.centroids[:, 0],self.centroids[:, 1],self.centroids[:, 2], c='red', label = "centroids", marker='+')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('before training phase')
            ax.legend()
            plt.show()
        
        
    def predict(self):
        
        for i in range(self.max_iter):
            self.clusters = [[] for _ in range(self.K)] 
            #on associe chaque point au centroid le plus proche grace a la fonction closest_centroid
            for sample in self.X:
                centroid_index = self.closest_centroid(sample, self.centroids)
                self.clusters[centroid_index].append(sample)
            

            #mise à jour des centroids
            centroidolds = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in self.clusters]
            
            for j,centroid in enumerate(self.centroids): #si un cluster est vide, le centroid reste le même
                if np.any((np.isnan(centroid))):
                    self.centroids[j] = centroidolds[j]

            self.centroids = np.array(self.centroids)

            if np.array_equal(centroidolds,self.centroids): #si les centroids de l'itération n et ceux de l'itération n-1 sont égaux, on sort de la boucle
                break
            else :
                self.plot(f"during training phase, step {i+1}")

        # training phase terminée, on affiche les résultat si on est en 2D ou 3D : 
        self.plot("after training phase")

        #on associe chaque point à l'indice du cluster

    def closest_centroid(self, sample, centroids):
        distances = [np.linalg.norm(sample-point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def plot(self, titre):
        self.clusters = [tab for tab in self.clusters if (len(tab)>0)] #supression des cluster vides

        if (self.features == 2) :

            for i in range(len(self.clusters)):
                plt.scatter(np.array(self.clusters[i])[:,0],np.array(self.clusters[i])[:,1], label = f"cluster {i+1}")
           
            plt.scatter(self.centroids[:, 0],self.centroids[:, 1], c='k', label = 'centroids', marker='1')
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title(titre)
            plt.legend()
            plt.show()
        elif (self.features == 3) : 
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            for i in range(len(self.clusters)):
            
                ax.scatter(np.array(self.clusters[i])[:,0],np.array(self.clusters[i])[:,1],np.array(self.clusters[i])[:,2], label = f"cluster {i+1}")
            
            ax.scatter(self.centroids[:, 0],self.centroids[:, 1],self.centroids[:, 2], c='k', label = "centroids", marker='+')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(titre)
            ax.legend()
            plt.show()
        

 
        




kmean1 = KMeans(3, 4, 100)
kmean1.predict()
            




        

