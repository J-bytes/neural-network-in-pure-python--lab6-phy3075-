# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 00:18:38 2020

@author: joeda
"""
import numpy as np

import matplotlib.pyplot as plt
from numba import prange,jit
#La retropropagation pour un reseau feed-forward avec une couche interne
#======================================================================
#variable globale
ni,nh,no    =13,10,1        # nombre d’unites d’entree, interne et de sortie
wih   =np.zeros([ni,nh])   # poids des connexions entree vers interne
who   =np.zeros([nh,no])   # poids des connexions interne vers sortie
ivec  =np.zeros(ni)        # signal en entree
sh    =np.zeros(nh)        # signal des neurones interne
so    =np.zeros(no)        # signal des neurones de sortie
err   =np.zeros(no)        # signal d’erreur des neurones de sortie
deltao=np.zeros(no)        # gradient d’erreur des neurones de sortie
deltah=np.zeros(nh)        # gradient d’erreur des neurones internes
eta   =0.1                 # parametre d’apprentissage
#----------------------------------------------------------------------
#Fonction d’activation sigmoidale
def actv(a):
    #return  1./(1.+np.exp(-a)) 
    return (np.tanh(a))**2         # Eq. (6.5)
#----------------------------------------------------------------------
#Derivee de la fonction d’activation sigmoidale
def dactv(s):
    return  s*(1.-s)                 # Eq. (6.19)
#----------------------------------------------------------------------
#fonction reseau feed-forward,calcul signal de sortie
def ffnn(ivec):
    ivectemp=np.array([ivec,]*nh).transpose()
          # couche d’entree a couche interne
    sh[:]=actv(np.sum(wih[:,0:no]*ivectemp[0:ni,:],axis=0))         # Eq. (6.2)
          # couche interne a couche de sortie
    shtemp=np.array([sh,]*no).transpose()
    so[:]=actv(np.sum(who[:,:]*shtemp[:,:],axis=0))            # Eq. (6.4)
    return 
#ENDfonctionffnn
#----------------------------------------------------------------------
#retropropagation du signal d’erreur et ajustement des poids du reseau
def backprop(err):
             # couche de sortie a couche interne
    deltao[:]=err[:]* dactv(so[:])      # Eq. (6.20)
        
    who[:,:]+=eta*deltao[:]*np.array([sh[:],]*no).transpose()   # Eq. (6.17) pour les wHO
    for ih in range(0,nh):           # couche interne a couche de sortie
        
       
        sum=np.sum(deltao[:]*who[ih,:])
        deltah[ih]=dactv(sh[ih])*sum           # Eq. (6.21)
        
        wih[:,ih]+=eta*deltah[ih]*ivec[:] # Eq. (6.17) pour les wIH
    return
#ENDfonctionbackpro

#LecodedelaFigure6.4doitprecedercequisuit
#----------------------------------------------------------------------
#fonctiondemelange
def randomize(n):
    dumvec=np.random.uniform(0,1,size=n) # tableau de nombre aleatoires
    return np.argsort(dumvec)        # retourne le tableau de rang
#ENDfonctionrandomize
#======================================================================
def retropropagation() :
    #MAIN:Entrainementd’un reseau par retropropagation
    nset  =50              # nombre de membres dans ensemble d’entrainement
    niter =100                # nombre d’iterations d’entrainement
    oset  =np.zeros([nset,no]) # sortie pour l’ensemble d’entrainement
    tset  =np.zeros([nset,ni]) # vecteurs-entree l’ensemble d’entrainement
    rmserr=np.zeros(niter)     # erreur rms d’entrainement
    
    #lecture/initialisationdel’ensembled’entrainement
   
    tset=np.loadtxt('trainingdata.txt',skiprows=1,usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12))[0:nset,:]                          # diverses instructions...
    oset[:,0]=np.loadtxt('trainingdata.txt',skiprows=1,usecols=(13))[0:nset]  
    for i in range(0,13) :
        tset[:,i]/=np.max(tset[:,i]) # normalisation
    
    #initialisationaleatoiredespoids
    wih[:,:]=np.random.uniform(-0.5,0.5,size=(ni,nh))
    who[:,:]=np.random.uniform(-0.5,0.5,size=(nh,no))
            
    for iter in range(0,niter):       # boucle sur les iteration d’entrainement
            sum=0
            print(iter)
            rvec=randomize(nset)           # melange des membres
            for itrain in range(0,nset):   # boucle sur l’ensemble d’entrainement
                itt=rvec[itrain]            # le membre choisi...
                ivec=tset[itt,:]            # ...et son vecteur d’entree
                ffnn(ivec)                  # calcule signal de sortie
                      # signaux d’erreur sur neurones de sortie
                err[:]=oset[itt]-so[:] #à changer [itt,:] si no>1
                          
            backprop(err)  # retropropagation
            sum+=np.sum(err[:]**2)
             
    #ENDbouclesurensemble
            
            rmserr[iter]=np.sqrt(sum/nset/no)   # erreur rms a cette iteration
            #ENDbouclesuriterationsd’entrainement
            
            #Maintenant laphase de test irait ci-dessous...
            #ENDMAIN

retropropagation()# pour l'entrainement 
#%% # pour la prédiction
#@jit(parallel=True)
def prediction() :
    tset=np.loadtxt('trainingdata.txt',skiprows=1) 
    nset=50 #nombre d'événement tester
    cheatsheet=tset[500:500+nset,13]
    tset=tset[500:500+nset,0:13]
    reponse=np.zeros(nset)
    for i in range(0,nset) :
        reponse[i]=ffnn(tset[i,:])
    return reponse,cheatsheet
 
        
reponse,cheatsheet=prediction()
plt.plot(np.abs(cheatsheet-reponse),'.')