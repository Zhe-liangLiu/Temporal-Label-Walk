import networkx as nx 
import copy 
import numpy as np 
from datetime import datetime, timedelta 
import os.path 
import math

#*********Computing label value about Eq(3)***********#
def ComputingLabelValue(data,G,alpha,c,beta):
        E=[]
        item=[]
        NODE=[]
        VALUE=[]
        N=list(G.nodes)
        N=np.array(N)
        for i in range(0,len(G.nodes)):
                VALUE.append([])
        tqian=data[0][0]
        for k in range(0,len(G.nodes)):
                a=np.zeros(len(G.nodes))
                a[k]=alpha
                E.append(a)
        for p in range(0,len(G.nodes)):
                a=np.zeros(len(G.nodes))
                a[p]=1
                NODE.append(a)
        for i in range(0,len(data)):
                thou=data[i][0]
                timew=thou-tqian
                item=[]
                for l in range(len(G.nodes)):
                        a=np.zeros(len(G.nodes))
                        item.append(a)
                for j in range(1,len(data[i])):
                        accept=np.where(N==data[i][j][1])[0][0]
                        send=np.where(N==data[i][j][0])[0][0]
                        item[accept]=item[accept]+(NODE[send]*beta*math.exp(-c*(timew.seconds))+E[send])
                for s in range(0,len(G.nodes)):
                        NODE[s]=NODE[s]*beta*math.exp(-c*(timew.seconds))+item[s]
                        q=np.linalg.norm(NODE[s])
                        if item[s].any()!=0:#Comment out these two lines for non-unitzation experiment#
                                NODE[s]=NODE[s]/q#Comment out these two lines for non-unitzation experiment#
                tqian=thou
        return NODE

#********Computing modularity of directed graph*********#
def DiModularity(division,DiG):
        M=len(DiG.edges)
        p=DiG._node.keys()
        q=list(p)
        N=list(DiG.nodes)
        N=np.array(N)
        com=[]
        A=nx.to_numpy_matrix(DiG)
        for a in range(0,len(q)):
                B=[]
                for b in range(0,len(q)):
                        if a==b:
                                B.append(0)
                        elif division[(np.where(N==q[a])[0][0])]==division[(np.where(N==q[b])[0][0])]:
                                B.append(1)
                        else:
                                B.append(0)
                com.append(B)
        SUM=0
        for v in range(0,len(A)):
                for u in range(0,len(A)):
                        kv=DiG.out_degree(q[v])
                        ku=DiG.in_degree(q[u])
                        SUM=SUM+(A[v,u]-(kv*ku)/M)*com[v][u]
        Q=SUM/M
        return Q

#*********Computing modularity of undirected graph for static community detection methods*********#
def Modularity(division,G):
        P=list(division)
        M=2*len(G.edges)
        label=np.zeros(len(G.nodes))
        p=G._node.keys()
        q=list(p)
        q=np.array(q)
        for r in range(0,len(P)):
                S=list(P[r])
                for k in range(0,len(S)):
                        index=np.where(q==S[k])
                        label[index[0][0]]=r
        com=[]
        for a in range(0,len(q)):
                B=[]
                for b in range(0,len(q)):
                        if a==b:
                                B.append(0)
                        elif label[a]==label[b]:
                                B.append(1)
                        else:
                                B.append(0)
                com.append(B)
        A=nx.to_numpy_matrix(G)
        SUM=0
        for v in range(0,len(A)):
                for u in range(0,len(A)):
                        kv=G.degree(q[v])
                        ku=G.degree(q[u])
                        SUM=SUM+(A[v,u]-(kv*ku)/M)*com[v][u]
        Q=SUM/M
        return Q

#*********Reading real data sets***********#
def Data2timedges(path):
        edgesTS=[]
        with open(path,'r') as fd:
                for line in fd.readlines():
                        line=line.strip()
                        #print('one',line)
                        items=line.split(' ')
                        #print(items)
                        tstamp=' '.join(items[0:2])
                        #print(tstamp)
                        tstamp=tstamp[1:-1]
                        tstamp = datetime.strptime(tstamp, '%Y-%m-%d %H:%M:%S')
                        #print(tstamp)
                        t=items[2:4]
                        #print(t)
                        t=map(int,t)
                        a=list(t)
                        #print(tuple(a))
                        edgesTS.append([tstamp, a])              
        fd.close()

        edges=[]
        sorted(edgesTS)
        T=[]
        T.append(edgesTS[0])
        for i in range(1,len(edgesTS)):
                if T[-1][0]<edgesTS[i][0]:
                        T.append(edgesTS[i])
                else:
                        T[-1].append(edgesTS[i][1])
        return edges,edgesTS,T
