

## Data

import numpy as np
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

C = Nearest(matrice_pos,matrice_dist,list_distrib)
col = ['red','blue','violet','black','green','magenta','orange','yellow','pink','cyan','brown']
for elem in list_distrib:
    for index in C[elem]:
        if (matrice_pos[index][2] == 'terminal'):
            plt.scatter(matrice_pos[index][0],matrice_pos[index][1],c = col[elem],marker="x")
        else :
            plt.scatter(matrice_pos[index][0],matrice_pos[index][1],c = col[elem],marker="o")
plt.show()

## 2 cluster in each previous cluster

def get_sub_matrix(cluster):
    mat = np.zeros((len(cluster),len(cluster)))
    centers_sorted = np.sort(cluster)
    for i in range(len(cluster)):
        for j in range(len(cluster)):
            mat[i][j] = matrice_dist[centers_sorted[i]][centers_sorted[j]]
    return mat

def get_k_sub_cluster(k,cluster):
    mat = get_sub_matrix(cluster)
    M1 , C1 = kMedoids(mat,k)
    for k in range(len(M1)):
        M1[k] = cluster[M1[k]]
    for k in range(len(C1)):
        for w in range(len(C1[k])):
            C1[k][w] = cluster[C1[k][w]]
    return M1, C1
    

M1 , C1 = get_k_sub_cluster(2,C[0])
## Test of medoids 

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

def kMedoids(D, k, tmax=200):
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

## Transforming mat_pos into an array

matrice_dist = np.asarray(matrice_dist)
    
import tsp
    
def chemin_principal(cluster):
    mat = np.zeros((len(cluster),len(cluster)))
    centers_sorted = np.sort(cluster)
    for i in range(len(cluster)):
        for j in range(len(cluster)):
            mat[i][j] = matrice_dist[centers_sorted[i]][centers_sorted[j]]
    r = range(len(mat))
    dist = {(i, j): mat[i][j] for i in r for j in r}
    A = tsp.tsp(r, dist)[1]
    B = []
    for k in range(len(A)):
        B.append(centers_sorted[A[k]])
    return B
    
    
fichier = open(filename +"/sortie.txt","w")
for i in range(len(W)):
    for j in range(len(W[i])-1):
        fichier.write(str(W[i][j])+" ")
    fichier.write(str(W[i][len(W[i])-1]))
    fichier.write("\n")
fichier.close()    
        
    

    