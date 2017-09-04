
# coding: utf-8

# In[1]:

import numpy as np
from sklearn.utils.extmath import randomized_svd as svd

# In[2]:

D = 512
ep = 0.1
NI = 1


# In[3]:

A = np.random.rand(D,D,2)
B = np.random.rand(D,D,2)
EAB = np.ones(D)
EBA = np.ones(D)
L = np.random.rand(D,D)
R = np.random.rand(D,D)


# In[4]:

H = np.reshape([[0.25,0,0,0],[0,-0.25,0.5,0],[0,0.5,-0.25,0],[0,0,0,0.25]],[2,2,2,2])
I = np.reshape(np.identity(4),[2,2,2,2])
expH = I - 4.*ep*H


# In[5]:

def d_max(t):
    return t/np.max(np.abs(t))


# In[6]:

def updateAB():
    global A,B,EAB,EBA,L,R
    EA = A * np.reshape(EBA,[D,1,1]) * np.reshape(EAB,[1,D,1])
    EB = B * np.reshape(EAB,[D,1,1]) * np.reshape(EBA,[1,D,1])
    REBA = 1./EBA
    AB = np.tensordot(EA,EB,[[1],[0]])
    HAB = np.tensordot(AB,expH,[[1,3],[0,1]])
    #U, S, V = np.linalg.svd(np.reshape(np.transpose(HAB,[0,2,1,3]),[2*D,2*D]))
    U, S, V = svd(np.reshape(np.transpose(HAB,[0,2,1,3]),[2*D,2*D]),D,n_iter=NI)
    ABnEAB = np.sqrt(S[:D])
    ABnA = np.transpose(np.reshape(U[:,:D],[D,2,D])*np.reshape(REBA,[D,1,1]),[0,2,1])
    ABnB = np.transpose(np.reshape(V[:D,:],[D,D,2])*np.reshape(REBA,[1,D,1]),[0,1,2])
    A, B, EAB = d_max(ABnA), d_max(ABnB), d_max(ABnEAB)


# In[7]:

def updateBA():
    global A,B,EAB,EBA,L,R
    EA = A * np.reshape(EBA,[D,1,1]) * np.reshape(EAB,[1,D,1])
    EB = B * np.reshape(EAB,[D,1,1]) * np.reshape(EBA,[1,D,1])
    REAB = 1./EAB
    BA = np.tensordot(EB,EA,[[1],[0]])
    HBA = np.tensordot(BA,expH,[[1,3],[0,1]])
    #U, S, V = np.linalg.svd(np.reshape(np.transpose(HBA,[0,2,1,3]),[2*D,2*D]))
    U, S, V = svd(np.reshape(np.transpose(HBA,[0,2,1,3]),[2*D,2*D]),D,n_iter=NI)
    BAnEBA = np.sqrt(S[:D])
    BAnB = np.transpose(np.reshape(U[:,:D],[D,2,D])*np.reshape(REAB,[D,1,1]),[0,2,1])
    BAnA = np.transpose(np.reshape(V[:D,:],[D,D,2])*np.reshape(REAB,[1,D,1]),[0,1,2])
    B, A, EBA = d_max(BAnB), d_max(BAnA), d_max(BAnEBA)


# In[8]:

def updateLR():
    global A,B,EAB,EBA,L,R
    EA = A * np.reshape(EBA,[D,1,1]) * np.reshape(EAB,[1,D,1])
    EB = B * np.reshape(EAB,[D,1,1]) * np.reshape(EBA,[1,D,1])
    LA = np.tensordot(np.tensordot(L,EA,[[1],[0]]),EA,[[0,2],[0,2]])
    LAB = np.tensordot(np.tensordot(LA,EB,[[1],[0]]),EB,[[0,2],[0,2]])
    BR = np.tensordot(np.tensordot(R,EB,[[1],[1]]),EB,[[0,2],[1,2]])
    ABR = np.tensordot(np.tensordot(BR,EA,[[1],[1]]),EA,[[0,2],[1,2]])
    L, R = d_max(LAB), d_max(ABR)


# In[9]:

def get_energy():
    global A,B,EAB,EBA,L,R
    EA = A * np.reshape(EBA,[D,1,1]) * np.reshape(EAB,[1,D,1])
    EB = B * np.reshape(EAB,[D,1,1]) * np.reshape(EBA,[1,D,1])
    LA = np.tensordot(L,EA,[[0],[0]])
    LAB = np.tensordot(LA,EB,[[1],[0]])
    LABR = np.tensordot(LAB,R,[[2],[0]])
    LABRA = np.tensordot(LABR,EA,[[0],[0]])
    LABRAB = np.tensordot(LABRA,EB,[[3,2],[0,1]])
    E_AB = np.tensordot(LABRAB,H,[[0,1,2,3],[0,1,2,3]])/np.tensordot(LABRAB,I,[[0,1,2,3],[0,1,2,3]])

    LABA = np.tensordot(LAB,EA,[[2],[0]])
    LABAA = np.tensordot(LABA,EA,[[0,1],[0,2]])
    LABAAB = np.tensordot(LABAA,EB,[[3],[0]])
    LABAABA = np.tensordot(LABAAB,EA,[[3],[0]])
    LABAABAB = np.tensordot(LABAABA,EB,[[4],[0]])
    LABAABABB = np.tensordot(LABAABAB,EB,[[1,6],[0,2]])
    LABAABABBR = np.tensordot(LABAABABB,R,[[5,4],[0,1]])
    E_BA = np.tensordot(LABAABABBR,H,[[0,1,2,3],[0,1,2,3]])/np.tensordot(LABAABABBR,I,[[0,1,2,3],[0,1,2,3]])

    E = (E_AB+E_BA)/2
    return E


# In[10]:

for _ in xrange(10):
    updateAB()
    updateBA()
    updateBA()
    updateAB()
for _ in xrange(10):
    updateLR()
print get_energy()


# In[ ]:



