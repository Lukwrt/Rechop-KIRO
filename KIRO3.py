

## Data

import numpy as np
import matplotlib.pyplot as plt
import csv

filename="/Users/samuel/Desktop/KIRO"
matrice_pos=[]
matrice_dist=[]
matrice_dist_temp=[]

with open(filename + "/nodes.csv", newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';', quotechar='|')
    for row in reader:
        matrice_pos.append(row)
del matrice_pos[0]
for i in range(0,len(matrice_pos)):
    matrice_pos[i][0]=float(matrice_pos[i][0])
    matrice_pos[i][1]=float(matrice_pos[i][1])

with open(filename + "/distances.csv", newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';', quotechar='|')
    for row in reader:
        matrice_dist_temp.append(float(row[0]))

matrice_dist=[]
n=len(matrice_pos)
for i in range(0,n):
    row=[]
    for j in range(0,n):
        row.append(matrice_dist_temp[i*n+j])
    matrice_dist.append(row)
    
    
def index(list,elem):
    for index in range(len(list)):
        if elem == list[index]:
            return index
            

list_distrib = []
for elem in matrice_pos:
    if (elem[2] == 'terminal'):
        plt.scatter(elem[0],elem[1],marker = 'x',c = 'blue')
        
    else :
        plt.scatter(elem[0],elem[1],marker = 'o',c = 'red')
        list_distrib.append(index(matrice_pos,elem))
plt.show()
    

def Nearest(matrix_pos,matrix_distance,list_distrib):
    C = {}
    for elem in list_distrib:
        C[elem] = []
        
    for index in range(len(matrix_pos)):
        dist = 1e10
        elem_mem = 0
        for elem in list_distrib:
            if (matrix_distance[index][elem] < dist):
                dist = matrix_distance[index][elem]
                elem_mem = elem
        C[elem_mem].append(index)
        
    for elem in list_distrib:
        C[elem] = np.array(C[elem])
    return C
    
   
## Cluster with distribution node in each one of them 

matrice_dist = np.asarray(matrice_dist)
matrice_dist_transp = np.zeros((matrice_dist.shape[1],matrice_dist.shape[0]))
for i in range(matrice_dist.shape[0]):
    for j in range(matrice_dist.shape[1]):
        matrice_dist_transp[j][i] = matrice_dist[i][j]


C = Nearest(matrice_pos,matrice_dist_transp,list_distrib)
col = ['red','blue','violet','black','green','magenta','orange','yellow','pink','cyan','brown']
for elem in list_distrib:
    for index in C[elem]:
        if (matrice_pos[index][2] == 'terminal'):
            plt.scatter(matrice_pos[index][0],matrice_pos[index][1],c = col[elem],marker="x")
        else :
            plt.scatter(matrice_pos[index][0],matrice_pos[index][1],c = col[elem],marker="o")
plt.show()


        
    
## Test of medoids 

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import sklearn.cluster as sk


# def kMedoids(D,n_clusters):
#     list = sk.SpectralClustering(n_clusters,assign_labels="discretize",random_state=0).fit(D).labels_
#     Clusters = {}
#     for index in range(n_clusters):
#         Clusters[index] = []
#     for w in range(len(list)):
#         for index in range(len(list_distrib)):
#             if list[w] == index:
#                 Clusters[index].append(w)
#     return Clusters
#     
def kMedoids(D, k, tmax=2000):
    # determine dimensions of distance matrix D
    m, n = D.shape
    # randomly initialize an array of k medoid indices
    M = np.sort(np.random.choice(n, k))
    
    # create a copy of the array of medoid indices
    Mnew = np.copy(M)
    
    # initialize a dictionary to represent clusters
    C = {}
    
    for t in range(tmax):
        # determine clusters, i.e. arrays of data indices
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
            
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            if (len(J) != 0):
                j = np.argmin(J)
                Mnew[kappa] = C[kappa][j]
        np.sort(Mnew) 
        
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
            
    # return results
    return M,C


## 2 cluster in each previous cluster

def get_sub_matrix(cluster,matrix_dist):
    mat = np.zeros((len(cluster),len(cluster)))
    centers_sorted = np.sort(cluster)
    for i in range(len(cluster)):
        for j in range(len(cluster)):
            mat[i][j] = matrix_dist[centers_sorted[i]][centers_sorted[j]]
    return mat

def get_k_sub_cluster(k,cluster,matrix_dist):
    mat = get_sub_matrix(cluster,matrix_dist)
    M1 ,C1 = kMedoids(mat,k)
    for k in range(len(C1)):
        for w in range(len(C1[k])):
            C1[k][w] = cluster[C1[k][w]]
    return C1
   
   
## Code a executer à la main avec C[0], C[1] etc ... et en faisant varier k
#7:  2 ou 3
#9 2 ou 3
#10 2 ou 3
k = 2
C1 = get_k_sub_cluster(k,C[10],matrice_dist_transp)
for k_ in range(k):
    print(len(C1[k_]))
    
for w in range(k):
    for j in C1[w]:
        plt.scatter(matrice_pos[j][0],matrice_pos[j][1],c = col[w])
        plt.annotate(j,xy =(matrice_pos[j][0],matrice_pos[j][1]))
plt.show()



## List of subclusters : 

def get_all_sub_clusters(C):
    Centers = []
    Clusters = {}
    for w in range(len(C)):
        test = False
        k = 1
        while(not test):
            k += 1
            M1, C1 = get_k_sub_cluster(k,C[w])
            for k_ in range(k):
                if len(C1[k_]) > 30:
                    test = True
        print(M1)
        test = False
        for k_ in range(k):
            Clusters[w + k_*len(C)] = C1[k_]
    return Centers, Clusters
        
        
            
#Centers, Clusters = get_all_sub_clusters(C)
        

    
 

## Transforming mat_pos into an array


    
import tsp
    
def chemin_principal(cluster,matrix_dist):
    mat = np.zeros((len(cluster),len(cluster)))
    centers_sorted = np.sort(cluster)
    for i in range(len(cluster)):
        for j in range(len(cluster)):
            mat[i][j] = matrix_dist[centers_sorted[i]][centers_sorted[j]]
    r = range(len(mat))
    dist = {(i, j): mat[i][j] for i in r for j in r}
    A = tsp.tsp(r, dist)[1]
    B = []
    for k in range(len(A)):
        B.append(centers_sorted[A[k]])
    return B
    

fichier = open(filename +"/sortie.txt","w")
for i in range(len(list_def_clust)):
    fichier.write("b ")
    for j in list_def_clust[i]: 
        fichier.write(str(j) +" ")
    fichier.write("\n")
fichier.close()    
        
    

    