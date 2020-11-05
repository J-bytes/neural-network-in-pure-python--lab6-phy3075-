# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 00:18:38 2020

@author: jonathan Beaulieu-Emond
"""


#La retropropagation pour un reseau feed-forward avec une couche interne
#======================================================================
import numpy as np
import math
import matplotlib.pyplot as plt
#Variables globales
def innitialisation() :
    #input,1,deep,2,output : structure du reseau désirée
   
    ni,n1,nd,n2,no    =13,4,4,4,1        # nombre d’unites d’entree, interne et de sortie
    depth=2
    wi1   =np.zeros([ni,n1])   # poids des connexions entree vers interne
    w1d   =np.zeros([n1,nd])   # poids des connexions interne vers sortie
    wdd=np.zeros((nd,nd,depth))
    wd2   =np.zeros([nd,n2])   # poids des connexions entree vers interne
    w2o   =np.zeros([n2,no])   # poids des connexions interne vers sortie
    ivec  =np.zeros(ni)        # signal en entree
    s1    =np.zeros(n1)        # signal des neurones interne
    sd    =np.zeros((nd,depth))        # signal des neurones interne
    s2    =np.zeros(n2)        # signal des neurones interne
    so    =np.zeros(no)        # signal des neurones de sortie
    err   =np.zeros(no)        # signal d’erreur des neurones de sortie
    deltao=np.zeros(no)        # gradient d’erreur des neurones de sortie
    delta1=np.zeros(n1)        # gradient d’erreur des neurones internes
    deltad=np.zeros((nd,depth))        # gradient d’erreur des neurones internes
    delta2=np.zeros(n2)        # gradient d’erreur des neurones internes
   
    eta   =0.5                 # parametre d’apprentissage
    param=(wi1,w1d,wdd,wd2,w2o,ni,n1,nd,n2,no,ivec,s1,sd,s2,so,err,deltao,delta1,deltad,delta2,eta,depth)
     #initialisationaleatoiredespoids
    
    wi1[:,:]=np.random.uniform(-0.5,0.5,size=(ni,n1))# poids entree-interne
    w1d[:,:]=np.random.uniform(-0.5,0.5,size=(n1,nd)) # poids interne-sortie
    wdd[:,:]=np.random.uniform(-0.5,0.5,size=(nd,nd,depth))# poids entree-interne
    wd2[:,:]=np.random.uniform(-0.5,0.5,size=(nd,n2)) # poids interne-sortie
    w2o[:,:]=np.random.uniform(-0.5,0.5,size=(n2,no)) # poids interne-sortie
    return param

def normalisation(data):
    return (data-data.mean(axis=0))/data.std(axis=0)
#----------------------------------------------------------------------
#Fonction d’activation sigmoidale#???
def actv(a):
    return  1./(1.+np.exp(-a))          # Eq. (6.5)
   # return (np.tanh(a))**2
#----------------------------------------------------------------------
#Derivee de la fonction d’activation sigmoidale
def dactv(s):
    return  s*(1.-s)                 # Eq. (6.19)
#----------------------------------------------------------------------
#fonction reseau feed-forward,calcul signal de sortie
def ffnn(param):
    wi1,w1d,wdd,wd2,w2o,ni,n1,nd,n2,no,ivec,s1,sd,s2,so,err,deltao,delta1,deltad,delta2,eta,depth=param
    s1[:]=actv( np.sum(wi1*ivec.reshape(ni,1),axis=0))        #
    # couche interne a couche de sortie
    s1temp=np.array([s1,]*nd).transpose()    
    # Eq. (6.3) avec b_1=0
    sd[:,0]=actv(np.sum(w1d[:,:]*s1temp[:,:],axis=0))              # Eq. (6.4)
    
    
    
    for i in range(1,depth) :
        sdtemp=np.array([sd[:,i-1],]*nd).transpose()
        sd[:,i]=actv(np.sum(wdd[:,:,i]*sdtemp[:,:],axis=0))
    
        
    sdeeptemp=np.array([sd[:,depth-1],]*n2).transpose()    
    s2[:]=actv(np.sum(wd2[:,:]*sdeeptemp[:,:],axis=0))
    
    s2temp=np.array([s2,]*no).transpose()
    so[:]=actv(np.sum(w2o[:,:]*s2temp[:,:],axis=0))
        
        
        
    return wi1,w1d,wdd,wd2,w2o,ni,n1,nd,n2,no,ivec,s1,sd,s2,so,err,deltao,delta1,deltad,delta2,eta,depth
#ENDfonctionffnn
#----------------------------------------------------------------------
#retropropagation du signal d’erreur et ajustement des poids du reseau
def backprop(param):
    wi1,w1d,wdd,wd2,w2o,ni,n1,nd,n2,no,ivec,s1,sd,s2,so,err,deltao,delta1,deltad,delta2,eta,depth=param

    #-----------------------------------------------
    deltao[:]=err[:]* dactv(so[:]) # Eq. (6.20)
  
    w2o[:,:]+=eta*np.outer(deltao[:],s2[:]).transpose() # Eq. (6.17) pour les wHO
             # couche interne a couche de sortie
    #------------------------------------------------ 
    sum=np.sum((deltao[:]*w2o[:,:]),axis=1)
    delta2[:]=dactv(s2[:])*sum           # Eq. (6.21)
        
    wd2[:,:]+=eta*np.outer(delta2[:],sd[:,depth-1]).transpose() # Eq. (6.17) pour les wIH pour les wIH
     #------------------------------------------------ 
    sum=np.sum((delta2[:]*wd2[:,:]),axis=1)
    deltad[:,depth-1]=dactv(sd[:,depth-1])*sum           # Eq. (6.21)
        
    wdd[:,:,depth-1]+=eta*np.outer(deltad[:,depth-1],sd[:,depth-2]).transpose() # Eq. (6.17) pour les wIH pour les wIH
     #------------------------------------------------ 
    for i in range(depth-2,0,-1) :
                 
        sum=np.sum((deltad[:,i+1]*wdd[:,:,i+1]),axis=1)
        deltad[:,i]=dactv(sd[:,i])*sum           # Eq. (6.21)
            
        wdd[:,:,i]+=eta*np.outer(deltad[:,i],sd[:,i-1]).transpose()
        
     #------------------------------------------------ 
    sum=np.sum((deltad[:,1]*wdd[:,:,0]),axis=1)
    deltad[:,0]=dactv(sd[:,0])*sum           # Eq. (6.21)
        
    w1d[:,:]+=eta*np.outer(deltad[:,0],s1[:]).transpose() # Eq. (6.17) pour les wIH pour les wIH
    #------------------------------------------------
    sum=np.sum((deltad[:,0]*w1d[:,:]),axis=1)
    delta1[:]=dactv(s1[:])*sum           # Eq. (6.21)
        
    wi1[:,:]+=eta*np.outer(delta1[:],ivec[:]).transpose() # Eq. (6.17) pour les wIH pour les wIH
    
    return wi1,w1d,wdd,wd2,w2o,ni,n1,nd,n2,no,ivec,s1,sd,s2,so,err,deltao,delta1,deltad,delta2,eta,depth
#ENDfonctionbackpro

#LecodedelaFigure6.4doitprecedercequisuit
#----------------------------------------------------------------------
#fonctiondemelange
def randomize(n):
    dumvec=np.zeros(n)
    for k in range(0,n):
        dumvec[k]=np.random.uniform() # tableau de nombre aleatoires
    return np.argsort(dumvec)        # retourne le tableau de rang
#ENDfonctionrandomize
#======================================================================
#MAIN:Entrainementd’unreseauparretropropagation
def training(param) :
    wi1,w1d,wdd,wd2,w2o,ni,n1,nd,n2,no,ivec,s1,sd,s2,so,err,deltao,delta1,deltad,delta2,eta,depth=param
    nset  =200                 # nombre de membres dans ensemble d’entrainement
    niter =300                # nombre d’iterations d’entrainement
    oset  =np.zeros([nset,no]) # sortie pour l’ensemble d’entrainement
    tset  =np.zeros([nset,ni]) # vecteurs-entree l’ensemble d’entrainement
    rmserr=np.zeros(niter)     # erreur rms d’entrainement
    
    #lecture/initialisationdel’ensembled’entrainement
    
    tset=np.loadtxt('trainingdata.txt',skiprows=1,usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12))[0:nset,:]                          # diverses instructions...
    oset[:,0]=np.loadtxt('trainingdata.txt',skiprows=1,usecols=(13))[0:nset]  
    tset=normalisation(tset)
    
          
     # Event ID inutile???                             # diverses instructions...
   
    
    for iter in range(0,niter):       # boucle sur les iteration d’entrainement
            sum=0.
            print(iter)
            rvec=randomize(nset)           # melange des membres
            for itrain in range(0,nset):   # boucle sur l’ensemble d’entrainement
                itt=rvec[itrain]            # le membre choisi...
                ivec=tset[itt,:]            # ...et son vecteur d’entree
                param=wi1,w1d,wdd,wd2,w2o,ni,n1,nd,n2,no,ivec,s1,sd,s2,so,err,deltao,delta1,deltad,delta2,eta,depth
                wi1,w1d,wdd,wd2,w2o,ni,n1,nd,n2,no,ivec,s1,sd,s2,so,err,deltao,delta1,deltad,delta2,eta,depth=ffnn(param) 
                
                err[:]=oset[itt,:]-so[:]
                sum+=np.sum(err[:]**2) # cumul pour calcul de l’erreur rms
                param=wi1,w1d,wdd,wd2,w2o,ni,n1,nd,n2,no,ivec,s1,sd,s2,so,err,deltao,delta1,deltad,delta2,eta,depth
                
                wi1,w1d,wdd,wd2,w2o,ni,n1,nd,n2,no,ivec,s1,sd,s2,so,err,deltao,delta1,deltad,delta2,eta,depth=backprop(param) # retropropagation          # retropropagation
    #ENDbouclesurensemble
            
            rmserr[iter]=math.sqrt(sum/nset/no)   # erreur rms a cette iteration
            #ENDbouclesuriterationsd’entrainement
        
            #Maintenantlaphasedetestiraitci-dessous...
            #ENDMAIN
    plt.plot(rmserr,'.')
    plt.show()
    return wi1,w1d,wdd,wd2,w2o,ni,n1,nd,n2,no,ivec,s1,sd,s2,so,err,deltao,delta1,deltad,delta2,eta,depth
   

def prediction(param) :
    wi1,w1d,wdd,wd2,w2o,ni,n1,nd,n2,no,ivec,s1,sd,s2,so,err,deltao,delta1,deltad,delta2,eta,depth=param
    tset=np.loadtxt('trainingdata.txt',skiprows=1,usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12))
    tset=normalisation(tset) # normalisation
    nset=500 #nombre d'événement tester
    cheatsheet=np.loadtxt('trainingdata.txt',skiprows=1,usecols=(13))[100:100+nset]
    tset=tset[1100:1100+nset,:]
    reponse=np.zeros(nset)
    for i in range(0,nset) :
        ivec=tset[i,:]
        wi1,w1d,wdd,wd2,w2o,ni,n1,nd,n2,no,ivec,s1,sd,s2,so,err,deltao,delta1,deltad,delta2,eta,depth=ffnn([wi1,w1d,wdd,wd2,w2o,ni,n1,nd,n2,no,ivec,s1,sd,s2,so,err,deltao,delta1,deltad,delta2,eta,depth])
        reponse[i]=so
    return reponse,cheatsheet


param=innitialisation()
param=training(param)
reponse,cheatsheet=prediction(param)
plt.plot(np.abs(cheatsheet-reponse),'.')
plt.show()


 
        
#reponse,cheatsheet=prediction()
#plt.plot(np.abs(cheatsheet-reponse),'.')