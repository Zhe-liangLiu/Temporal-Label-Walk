import networkx as nx 
import copy 
import numpy as np 
from datetime import datetime, timedelta 
import os.path 
import matplotlib.pyplot as plt
import math
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.cluster import MiniBatchKMeans
import time
from networkx.algorithms import community
from sklearn import metrics
import Function

nodecolor=[]
nodesize=[]
path='E:/temporal networks/sday.txt'
edges,edgesTS,T=Function.Data2timedges(path)
G=nx.DiGraph()
for item in edgesTS:
        edges.append(item[1])
G.add_edges_from(edges)
N=list(G.nodes)
N=np.array(N)
p=G._node.keys()
q=list(p)
#****** Experiment about Table1 & Table3 ******#
NODE=Function.ComputingLabelValue(T,G,1,0.000001,1)
start=time.clock()
#AP method
model=AffinityPropagation(damping=0.5,max_iter=1000,convergence_iter=7200,copy=True,affinity='euclidean').fit(NODE)
#DBSCAN method
model=DBSCAN(eps=0.6,min_samples=2,metric='manhattan',metric_params=None,algorithm='ball_tree',leaf_size=50,p=3,n_jobs=1).fit(NODE)
#Mini Batch K-Means method
model=MiniBatchKMeans(n_clusters=14,init='k-means++',max_iter=110,batch_size=210,verbose=0,compute_labels=True,random_state=None,tol=0.0,max_no_improvement=10,init_size=None,n_init=10).fit(NODE)
end=time.clock()
print(str(end-start))
C=model.labels_
Q=Function.DiModularity(C,G)
#******End******#

#****** Experiment about Fig.6 ******#
for b in range(0,len(q)):
        a=q[b]
        if C[(np.where(N==a)[0][0])]==C[(np.where(N==3)[0][0])]:
                nodecolor.append('coral')
                nodesize.append(200)
        elif C[(np.where(N==a)[0][0])]==C[np.where(N==1713)[0][0]]:
                nodecolor.append('dodgerblue')
                nodesize.append(200)
        else:
                nodecolor.append('black')
                nodesize.append(100)
nx.draw(G,pos=nx.circular_layout(G),with_labels=False,font_size=11,node_size=nodesize,node_color=nodecolor,edge_color='gainsboro',linewidths=1,width=1.0,edgecolors='black')
plt.show()
#******End******#

#****** Experiment about Table2 ******#
G=nx.Graph()
for item in edgesTS:
        edges.append(item[1])
G.add_edges_from(edges)
#LPA mehod
division=community.label_propagation.label_propagation_communities(G)
#K-clique
division=community.k_clique_communities(G,2)
#Asynchronous
division=community.label_propagation.asyn_lpa_communities(G)
Q=Function.Modularity(division,G)
#******End******#

#****** Experiment about Fig.8 ******#
for i in range(0,4):       
        strings=['o','v','x','^','*']
        Label=['K-clique','Asynchronous','LPA','TLW']
        x=['1 day','1 week','1 month','2 month','3 month']
        y=[[0.78,0.088,0.051,0.036,0.038],[0.78,0.057,0.045,0.035,0.032],[0.78,0.528,0.205,0.063,0.088],[0.773,0.408,0.311,0.213,0.225]]
        plt.plot(x,y[i],marker=strings[i],label=Label[i])
        plt.legend(loc='best',prop={'size':13})
plt.show()
#******End******#
