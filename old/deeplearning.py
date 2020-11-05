# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 00:18:38 2020

#@author: joeda
"""


#La retropropagation pour un reseau feed-forward avec une couche interne
#======================================================================
import numpy as np
import math
import matplotlib.pyplot as plt
from numba import jit,njit
#Variables globales
def setup() :
   
    depth=4
    #               
    ni,nh,nn,no    =13,8,8,1   # nombre d’unites d’entree, interne et de sortie
    wih   =np.zeros([ni,nh])   # poids des connexions entree vers interne
    whn   =np.zeros((nh,nn))
    whdeep=np.zeros((nn,nn,depth))
    wno   =np.zeros([nn,no])   # poids des connexions interne vers sortie
    
    #On inclus des biais sur nos connexions
    bih   =np.zeros([ni,nh])
    whn   =np.zeros((nh,nn))
    whdeep=np.zeros((nn,nn,depth))
    wno   =np.zeros([nn,no])   # poids des connexions interne vers sortie
    
    
    ivec  =np.zeros(ni)        # signal en entree
    sh    =np.zeros(nh)        # signal des neurones interne
    sn    =np.zeros(nn)
    sdeep=np.zeros((nn,depth))
    so    =np.zeros(no)        # signal des neurones de sortie
    err   =np.zeros(no)        # signal d’erreur des neurones de sortie
    deltao=np.zeros(no)        # gradient d’erreur des neurones de sortie
    deltan=np.zeros(nn)
    deltah=np.zeros(nh)        # gradient d’erreur des neurones internes
    deltadeep=np.zeros((nn,depth))
    eta   =0.1                 # parametre d’apprentissage
    return wih,wno,ni,no,ivec,sh,so,err,deltao,deltah,eta,nh,deltan,whn,sn,nn,sdeep,whdeep,deltadeep,depth
#----------------------------------------------------------------------
#Fonction d’activation sigmoidale#???
#@jit
def sigmoide(a):
    return  1./(1.+np.exp(-a))          # Eq. (6.5)
   # return (np.tanh(a))**2

#Derivee de la fonction d’activation sigmoidale
#@jit
def dsigmoide(s):
    #s=actv(s)
    return  s*(1.-s)                 # Eq. (6.19)


#@jit
def tanhyper(a) :
    return np.tanh(a)
#@jit
def dtanhyper(a) :
    return 1-(np.tanh(a))**2
#@njit
def swish(a) :
    return  a/(1.+np.exp(-a))
#@njit
def dswish(a) :
    return swish(a)+sigmoide(a)*(1-swish(a))
#@jit
def normalisation(data):
    return (data-data.mean(axis=0))/data.std(axis=0)
#----------------------------------------------------------------------
#fonction reseau feed-forward,calcul signal de sortie
#@jit
def ffnn(wih,wno,ni,no,ivec,sh,so,err,deltao,deltah,eta,nh,deltan,whn,sn,nn,sdeep,whdeep,deltadeep,depth,actv):
           # couche d’entree a couche interne

     
    for ih in range(0,nh):           # couche d’entree a couche interne
        sh[ih]=actv( np.sum(wih[:,ih]*ivec[:]))        # Eq. (6.1) 
           
    #couche interne à couche interne 
    shtemp=np.array([sh,]*nn)
    sdeep[:,0]=actv(np.sum(whdeep[:,:,0]*shtemp[:,:],axis=0))
    for i in range(1,depth) :
        sdeeptemp=np.array([sdeep[:,i-1],]*nn)
        sdeep[:,i]=actv(np.sum(whdeep[:,:,i]*sdeeptemp,axis=0))
    
    # couche interne a couche de sortie
    sdeeptemp=np.array([sdeep[:,depth-1],]*nn)
    sn[:]=actv(np.sum(whdeep[:,:,depth-1]*sdeeptemp,axis=0))
    # Eq. (6.3) avec b_1=0
    
    so[:]=actv(np.sum((wno[:,:]*sn[:].reshape(nn,no)),axis=0))              # Eq. (6.4)
    return 
#ENDfonctionffnn
#----------------------------------------------------------------------
#retropropagation du signal d’erreur et ajustement des poids du reseau
#@jit
def backprop(wih,wno,ni,no,ivec,sh,so,err,deltao,deltah,eta,nh,deltan,whn,sn,nn,sdeep,whdeep,deltadeep,depth,dactv):
    """
    deltao[:]=err[:]* dactv(so[:])      # Eq. (6.20)
        
    who[:,:]+=eta*deltao[:]*np.array([sh[:],]*no).transpose()   # Eq. (6.17) pour les wHOpour les wHO
    """
   # couche de sortie a couche interne
    deltao[:]=err[:]* dactv(so[:]) # Eq. (6.20)
       
    wno[:,:]+=eta*np.dot(deltao[:],sn[:].reshape(1,nn)).reshape(nn,no) # Eq. (6.17) pour les wHO
    
    #couche interne à couche interne
    sum=np.sum(deltao[:]*wno[:,:],axis=0)
    deltan[:]=dactv(sn[:])*sum           # Eq. (6.21)
        
    whn[:,:]+=eta*np.dot(deltan[:].reshape(nn,1),sh[:].reshape(1,nh)).reshape(nh,nn) # Eq. (6.17) pour les wIH pour les wIH
    
    #Couche du deep network
    sum=np.sum(deltan[:]*whdeep[:,:,depth-1],axis=0)
    deltadeep[:,depth-1]=dactv(sdeep[:,depth-1])*sum 
    whdeep[:,:,depth-1]+=eta*np.dot(deltadeep[:,depth-1].reshape(nn,1),sdeep[:,depth-1].reshape(1,nn))#reshape a corriger
    for i in range(depth-2,0,-1) :
        sum=np.sum(deltadeep[:,i+1]*whdeep[:,:,i+1],axis=0)
        deltadeep[:,i]=dactv(sdeep[:,i])*sum 
        whdeep[:,:,i]+=eta*np.dot(deltadeep[:,i].reshape(nn,1),sdeep[:,i+1].reshape(1,nn))#reshape a corriger
    
    # couche deep network a couche d'entrée
    
    sum=np.sum(deltadeep[:,0]*whdeep[:,:,0],axis=1)
    deltah[:]=dactv(sh[:])*sum           # Eq. (6.21)
        
    wih[:,:]+=eta*np.dot(deltah[:].reshape(nh,1),ivec[:].reshape(1,ni)).reshape(ni,nh) # Eq. (6.17) pour les wIH pour les wIH
    return 
#ENDfonctionbackpro

#LecodedelaFigure6.4doitprecedercequisuit
#----------------------------------------------------------------------
#fonctiondemelange
#@jit
def randomize(n):
    dumvec=np.zeros(n)
    for k in range(0,n):
        dumvec[k]=np.random.uniform() # tableau de nombre aleatoires
    return np.argsort(dumvec)        # retourne le tableau de rang
#ENDfonctionrandomize
#======================================================================
#MAIN:Entrainementd’unreseauparretropropagation
#@jit
def training(actv,dactv,wih,wno,ni,no,ivec,sh,so,err,deltao,deltah,eta,nh,deltan,whn,sn,nn,sdeep,whdeep,deltadeep,depth) :
    
    nset  =200                 # nombre de membres dans ensemble d’entrainement
    niter =1000               # nombre d’iterations d’entrainement
    oset  =np.zeros([nset,no]) # sortie pour l’ensemble d’entrainement
    tset  =np.zeros([nset,ni]) # vecteurs-entree l’ensemble d’entrainement
    rmserr=np.zeros(niter)     # erreur rms d’entrainement
    
    #lecture/initialisationdel’ensembled’entrainement
    
    tset=np.loadtxt('trainingdata.txt',skiprows=1,usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12))[0:nset,:]                          # diverses instructions...
    oset[:,0]=np.loadtxt('trainingdata.txt',skiprows=1,usecols=(13))[0:nset]  
    tset=normalisation(tset)
    
          
     # Event ID inutile???                             # diverses instructions...
    #initialisationaleatoiredespoids
    
    wih[:,:]=np.random.uniform(-0.5,0.5,size=(ni,nh))# poids entree-interne
    whn[:,:]=np.random.uniform(-0.5,0.5,size=(nh,nn)) # poids interne-sortie
    wno[:,:]=np.random.uniform(-0.5,0.5,size=(nn,no)) # poids interne-sortie
    whdeep[:,:]=np.random.uniform(-0.5,0.5,size=(nn,nn,depth))
   
    for iter in range(0,niter):       # boucle sur les iteration d’entrainement
            sum=0.
            print(iter)
            rvec=randomize(nset)           # melange des membres
            for itrain in range(0,nset):   # boucle sur l’ensemble d’entrainement
                itt=rvec[itrain]            # le membre choisi...
                ivec=tset[itt,:]            # ...et son vecteur d’entree
                ffnn(wih,wno,ni,no,ivec,sh,so,err,deltao,deltah,eta,nh,deltan,whn,sn,nn,sdeep,whdeep,deltadeep,depth,actv) 
               
                err[:]=oset[itt,:]-so[:]
                sum+=np.sum(err[:]**2) # cumul pour calcul de l’erreur rms
                backprop(wih,wno,ni,no,ivec,sh,so,err,deltao,deltah,eta,nh,deltan,whn,sn,nn,sdeep,whdeep,deltadeep,depth,dactv) # retropropagation          # retropropagation
    #ENDbouclesurensemble
            
            rmserr[iter]=math.sqrt(sum/nset/no)   # erreur rms a cette iteration
            #ENDbouclesuriterationsd’entrainement
        
            #Maintenantlaphasedetestiraitci-dessous...
            #ENDMAIN
    plt.plot(rmserr,'.',label='activation='+str(actv))
   

def prediction() :
    tset=np.loadtxt('trainingdata.txt',skiprows=1,usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12))
    tset=normalisation(tset)
    nset=50 #nombre d'événement tester
    cheatsheet=np.loadtxt('trainingdata.txt',skiprows=1,usecols=(13))[100:100+nset]
    tset=tset[100:100+nset,:]
    reponse=np.zeros(nset)
    for i in range(0,nset) :
      ffnn(tset[i,:],nh,swish)
      reponse[i]=so
    return reponse,cheatsheet
"""
for actv,dactv in zip([sigmoide,tanhyper,swish],[dsigmoide,dtanhyper,dswish]) :
    setup()
    training(actv,dactv)
    reponse,cheatsheet=prediction()
    reponse2=np.round(reponse,0)
    reponse2[np.where(reponse2>1)]=1
    popt=np.where(reponse==cheatsheet)[0]
    print(len(popt))
"""
wih,wno,ni,no,ivec,sh,so,err,deltao,deltah,eta,nh,deltan,whn,sn,nn,sdeep,whdeep,deltadeep,depth=setup()
actv,dactv=sigmoide,dsigmoide
training(actv,dactv,wih,wno,ni,no,ivec,sh,so,err,deltao,deltah,eta,nh,deltan,whn,sn,nn,sdeep,whdeep,deltadeep,depth)
reponse,cheatsheet=prediction()
reponse2=np.round(reponse,0)
reponse2[np.where(reponse2>1)]=1
popt=np.where(reponse==cheatsheet)[0]
print(len(popt))
plt.legend()
plt.xlabel('iteration')
plt.ylabel('Erreurs')

 
        
reponse,cheatsheet=prediction()
popt=np.where(np.round(reponse,0)==cheatsheet)[0]
print(len(popt))
#plt.plot(np.abs(cheatsheet-reponse),'.')


