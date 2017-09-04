
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np


# In[2]:

D = 512
ep = 0.1


# In[3]:

def model(A,B,EAB,EBA,L,R):
    
    with tf.name_scope("constant"):
        H = tf.reshape(tf.constant([[0.25,0,0,0],[0,-0.25,0.5,0],[0,0.5,-0.25,0],[0,0,0,0.25]],
                                   dtype=tf.float32),[2,2,2,2],name="Hamiltonian")
        I = tf.reshape(tf.constant([[1.,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                                   dtype=tf.float32),[2,2,2,2],name="Identity")
        expH = I - 4. * ep * H

    with tf.name_scope("parameter"):
        REAB = tf.reciprocal(EAB,name="REAB")
        REBA = tf.reciprocal(EBA,name="REBA")

        EA = tf.multiply(tf.multiply(A,tf.reshape(EBA,[D,1,1])),tf.reshape(EAB,[1,D,1]),name="EA")
        EB = tf.multiply(tf.multiply(B,tf.reshape(EAB,[D,1,1])),tf.reshape(EBA,[1,D,1]),name="EB")

        AB = tf.tensordot(EA,EB,[[1],[0]],name="AB")
        BA = tf.tensordot(EB,EA,[[1],[0]],name="BA")

    def d_max(t):
        with tf.name_scope("d_max"):
            return t/tf.reduce_max(tf.abs(t))

    with tf.name_scope("updateAB"):
        HAB = tf.tensordot(AB,expH,[[1,3],[0,1]],name="HAB")
        S, U, V = tf.svd(tf.reshape(tf.transpose(HAB,[0,2,1,3]),[2*D,2*D]))
        ABnEAB = tf.sqrt(S[:D],name="nE")
        ABnA = tf.transpose(tf.multiply(tf.reshape(U[:,:D],[D,2,D]),tf.reshape(REBA,[D,1,1])),[0,2,1],name="nA")
        ABnB = tf.transpose(tf.multiply(tf.reshape(V[:,:D],[D,2,D]),tf.reshape(REBA,[D,1,1])),[2,0,1],name="nB")

    with tf.name_scope("updateBA"):
        HBA = tf.tensordot(BA,expH,[[1,3],[0,1]],name="HBA")
        S, U, V = tf.svd(tf.reshape(tf.transpose(HBA,[0,2,1,3]),[2*D,2*D]))
        BAnEBA = tf.sqrt(S[:D],name="nE")
        BAnB = tf.transpose(tf.multiply(tf.reshape(U[:,:D],[D,2,D]),tf.reshape(REAB,[D,1,1])),[0,2,1],name="nB")
        BAnA = tf.transpose(tf.multiply(tf.reshape(V[:,:D],[D,2,D]),tf.reshape(REAB,[D,1,1])),[2,0,1],name="nA")

    with tf.name_scope("updateL"):
        LA = tf.tensordot(tf.tensordot(L,EA,[[1],[0]]),EA,[[0,2],[0,2]],name="LA")
        LAB = tf.tensordot(tf.tensordot(LA,EB,[[1],[0]]),EB,[[0,2],[0,2]],name="LAB")

    with tf.name_scope("updateR"):
        BR = tf.tensordot(tf.tensordot(R,EB,[[1],[1]]),EB,[[0,2],[1,2]],name="BR")
        ABR = tf.tensordot(tf.tensordot(BR,EA,[[1],[1]]),EA,[[0,2],[1,2]],name="ABR")

    with tf.name_scope("Energy"):
        LA = tf.tensordot(L,EA,[[0],[0]])
        _LAB = tf.tensordot(LA,EB,[[1],[0]])
        LABR = tf.tensordot(_LAB,R,[[2],[0]])
        LABRA = tf.tensordot(LABR,EA,[[0],[0]])
        LABRAB = tf.tensordot(LABRA,EB,[[3,2],[0,1]])
        E_AB = tf.tensordot(LABRAB,H,[[0,1,2,3],[0,1,2,3]])/tf.tensordot(LABRAB,I,[[0,1,2,3],[0,1,2,3]])

        LABA = tf.tensordot(_LAB,EA,[[2],[0]])
        LABAA = tf.tensordot(LABA,EA,[[0,1],[0,2]])
        LABAAB = tf.tensordot(LABAA,EB,[[3],[0]])
        LABAABA = tf.tensordot(LABAAB,EA,[[3],[0]])
        LABAABAB = tf.tensordot(LABAABA,EB,[[4],[0]])
        LABAABABB = tf.tensordot(LABAABAB,EB,[[1,6],[0,2]])
        LABAABABBR = tf.tensordot(LABAABABB,R,[[5,4],[0,1]])
        E_BA = tf.tensordot(LABAABABBR,H,[[0,1,2,3],[0,1,2,3]])/tf.tensordot(LABAABABBR,I,[[0,1,2,3],[0,1,2,3]])

        E = (E_AB+E_BA)/2
    
    with tf.name_scope("d_max"):
        return d_max(ABnA), d_max(ABnB), d_max(ABnEAB), d_max(BAnA), d_max(BAnB), d_max(BAnEBA), d_max(LAB), d_max(ABR), E


# In[4]:

def for_loop_wrap(f):
    def __tmp__(i,a):
        with tf.name_scope("inc"):
            i = i+1
        return i,f(*a)
    return __tmp__

def loop_n(n):
    def cond(i,*a):
        with tf.name_scope("cond"):
            return i<n
    return cond

@for_loop_wrap
def updateABBA(A,B,EAB,EBA,L,R):
    with tf.name_scope("1AB"):
        A, B, EAB, _, _, _, _, _, _ = model(A,B,EAB,EBA,L,R)
    with tf.name_scope("2BA"):
        _, _, _, A, B, EBA, _, _, _ = model(A,B,EAB,EBA,L,R)
    with tf.name_scope("3BA"):
        _, _, _, A, B, EBA, _, _, _ = model(A,B,EAB,EBA,L,R)
    with tf.name_scope("4AB"):
        A, B, EAB, _, _, _, _, _, _ = model(A,B,EAB,EBA,L,R)
    return A,B,EAB,EBA,L,R

@for_loop_wrap
def updateLR(A,B,EAB,EBA,L,R):
    with tf.name_scope("updateLR"):
        _, _, _, _, _, _, L, R, _ = model(A,B,EAB,EBA,L,R)
    return A,B,EAB,EBA,L,R

def get_energy(A,B,EAB,EBA,L,R):
    with tf.name_scope("get_energy"):
        _, _, _, _, _, _, _, _, E = model(A,B,EAB,EBA,L,R)
    return E


# In[5]:

with tf.name_scope("init"):
    A = tf.random_normal(shape=[D, D, 2], dtype=tf.float32,name="A")
    B = tf.random_normal(shape=[D, D, 2], dtype=tf.float32, name="B")
    EAB = tf.ones(shape=[D], dtype=tf.float32, name="EAB")
    EBA = tf.ones(shape=[D], dtype=tf.float32, name="EBA")
    L = tf.random_normal(shape=[D, D], dtype=tf.float32, name="L")
    R = tf.random_normal(shape=[D, D], dtype=tf.float32, name="R")
    data = A,B,EAB,EBA,L,R


# In[6]:

_, data = tf.while_loop(loop_n(10),updateABBA,(tf.constant(0),data),name="updateABBA_loop")
_, data = tf.while_loop(loop_n(10),updateLR,(tf.constant(0),data),name="updateLR_loop")
E = get_energy(*data)


# In[7]:

config = tf.ConfigProto()
config.device_count["GPU"] = 1
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())


# In[8]:

get_ipython().run_cell_magic(u'time', u'', u'sess.run(E)')


# In[9]:

# fw = tf.summary.FileWriter(".",tf.get_default_graph())


# fortran : Wall 14.7s, User 79.3s
# 
# cpu     : Wall 22.6s, User 43.1s
# 
# gpu     : Wall 16.6s, User 15.9s
# 
# np      : Wall 17.1s, User 68.3s
