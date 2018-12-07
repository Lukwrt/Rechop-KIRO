import numpy as np
import matplotlib.pyplot as plt
import csv

filename="/Users/samuel/Desktop/KIRO/Paris"
matrice_pos=[]
matrice_dist=[]
matrice_dist_temp = []


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

matrice_dist = np.asarray(matrice_dist)
matrice_dist_transp = np.zeros((matrice_dist.shape[1],matrice_dist.shape[0]))
for i in range(matrice_dist.shape[0]):
    for j in range(matrice_dist.shape[1]):
        matrice_dist_transp[j][i] = matrice_dist[i][j]
        
    
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
    
C = Nearest(matrice_pos,matrice_dist_transp,list_distrib)
col = ['red','blue','violet','black','green','magenta','orange','yellow','pink','cyan','brown']
for elem in list_distrib:
    for index in C[elem]:
        if (matrice_pos[index][2] == 'terminal'):
            plt.scatter(matrice_pos[index][0],matrice_pos[index][1],c = col[elem],marker="x")
        else :
            plt.scatter(matrice_pos[index][0],matrice_pos[index][1],c = col[elem],marker="o")
plt.show()


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
    
cluster = [i for i in range(len(matrice_pos))]


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
    
    
def get_all_clusters(list_cluster):
    list_all_clust =[]
    for i in range(len(list_cluster)):
        k = 0
        bool = True
        while bool:
            bool = False
            k += 1 
            C1 = get_k_sub_cluster(k,list_cluster[i],matrice_dist)
            for w in range(len(C1)):
                if len(C1[w])>30:
                    bool = True
        for w in range(len(C1)):
            if i not in C1[w]:
                C1[w] = np.insert(C1[w],0,i)
        for w in range(len(C1)):
            list_all_clust.append(chemin_principal(C1[w],matrice_dist))
        
    return list_all_clust
                

# 
# k = 3
# C1 = get_k_sub_cluster(k,C[1],matrice_dist)
# for k_ in range(k):
#     print(len(C1[k_]))
#     
# for w in range(k):
#     for j in C1[w]:
#         plt.scatter(matrice_pos[j][0],matrice_pos[j][1],c = col[w])
#         plt.annotate(j,xy =(matrice_pos[j][0],matrice_pos[j][1]))
# plt.show()

def Cost(_list_cluster):
    s = 0 
    for w in range(len(_list_cluster)):
        for k in range(len(_list_cluster[w])-1):
            s+= matrice_dist[_list_cluster[w][k]][_list_cluster[w][k+1]]
        s+= matrice_dist[_list_cluster[w][len(_list_cluster[w])-1]][_list_cluster[w][0]]
    return s


min = 1e8
list_memo = []
for k in range(3):
    print(k)
    list_cluster = get_all_clusters(C)
    a = Cost(list_cluster)
    if a < min :
        list_memo = list_cluster
        print(a)
        fichier = open(filename +"/sortie_"+"Paris"+".txt","w")
        for i in range(len(list_cluster)-1):
            fichier.write("b ")
            for j in range(len(list_cluster[i])-1): 
                fichier.write(str(list_cluster[i][j]) +" ")
            fichier.write(str(list_cluster[i][len(list_cluster[i])-1]))
            fichier.write("\n")
        fichier.write("b ")
        for j in range(len(list_cluster[len(list_cluster)-1])-1): 
            fichier.write(str(list_cluster[len(list_cluster)-1][j]) +" ")
        fichier.write(str(list_cluster[len(list_cluster)-1][len(list_cluster[len(list_cluster)-1])-1]))
        fichier.close()
        
        min = a



    
    
for w in range(len(list_memo)):
    col=np.random.rand(3,)
    for k in range(len(list_memo[w])-1):
        plt.plot([matrice_pos[list_memo[w][k]][0],matrice_pos[list_memo[w][k+1]][0]],[matrice_pos[list_memo[w][k]][1],matrice_pos[list_memo[w][k+1]][1]],c=col )
    plt.plot([matrice_pos[list_memo[w][len(list_memo[w])-1]][0],matrice_pos[list_memo[w][0]][0]],[matrice_pos[list_memo[w][len(list_memo[w])-1]][1],matrice_pos[list_memo[w][0]][1]],c=col)
plt.show()

        