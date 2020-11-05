# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 00:18:38 2020

@author: joeda
"""


#La retropropagation pour un reseau feed-forward avec une couche interne
#======================================================================
import numpy as np
import math
import matplotlib.pyplot as plt
#Variables globales
def innitialisation() :
   
   
    ni,nh,no    =13,8,1        # nombre d’unites d’entree, interne et de sortie
    wih   =np.zeros([ni,nh])   # poids des connexions entree vers interne
    who   =np.zeros([nh,no])   # poids des connexions interne vers sortie
    ivec  =np.zeros(ni)        # signal en entree
    sh    =np.zeros(nh)        # signal des neurones interne
    so    =np.zeros(no)        # signal des neurones de sortie
    err   =np.zeros(no)        # signal d’erreur des neurones de sortie
    deltao=np.zeros(no)        # gradient d’erreur des neurones de sortie
    deltah=np.zeros(nh)        # gradient d’erreur des neurones internes
    eta   =0.1                 # parametre d’apprentissage
    param=(wih,who,ni,no,ivec,sh,so,err,deltao,deltah,eta,nh)
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
    wih,who,ni,no,ivec,sh,so,err,deltao,deltah,eta,nh=param
    sh[:]=actv( np.sum(wih*ivec.reshape(ni,1),axis=0))        #
    # couche interne a couche de sortie
    shtemp=np.array([sh,]*no).transpose()    
    # Eq. (6.3) avec b_1=0
    so[:]=actv(np.sum(who[:,:]*shtemp[:,:],axis=0))              # Eq. (6.4)
    return wih,who,ni,no,ivec,sh,so,err,deltao,deltah,eta,nh
#ENDfonctionffnn
#----------------------------------------------------------------------
#retropropagation du signal d’erreur et ajustement des poids du reseau
def backprop(param):
    wih,who,ni,no,ivec,sh,so,err,deltao,deltah,eta,nh=param
    """
    deltao[:]=err[:]* dactv(so[:])      # Eq. (6.20)
        
    who[:,:]+=eta*deltao[:]*np.array([sh[:],]*no).transpose()   # Eq. (6.17) pour les wHOpour les wHO
    """
    #for io in range(0,no): # couche de sortie a couche interne
    deltao[:]=err[:]* dactv(so[:]) # Eq. (6.20)
  
    who[:,:]+=eta*np.outer(deltao[:],sh[:]).transpose() # Eq. (6.17) pour les wHO
             # couche interne a couche de sortie
        
    sum=np.sum((deltao[:]*who[:,:]),axis=1)
    deltah[:]=dactv(sh[:])*sum           # Eq. (6.21)
        
    wih[:,:]+=eta*np.outer(deltah[:],ivec[:]).transpose() # Eq. (6.17) pour les wIH pour les wIH
    return wih,who,ni,no,ivec,sh,so,err,deltao,deltah,eta,nh
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
    wih,who,ni,no,ivec,sh,so,err,deltao,deltah,eta,nh=param
    nset  =200                 # nombre de membres dans ensemble d’entrainement
    niter =200                # nombre d’iterations d’entrainement
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
    who[:,:]=np.random.uniform(-0.5,0.5,size=(nh,no)) # poids interne-sortie
    
    for iter in range(0,niter):       # boucle sur les iteration d’entrainement
            sum=0.
            print(iter)
            rvec=randomize(nset)           # melange des membres
            for itrain in range(0,nset):   # boucle sur l’ensemble d’entrainement
                itt=rvec[itrain]            # le membre choisi...
                ivec=tset[itt,:]            # ...et son vecteur d’entree
                param=wih,who,ni,no,ivec,sh,so,err,deltao,deltah,eta,nh
                wih,who,ni,no,ivec,sh,so,err,deltao,deltah,eta,nh=ffnn(param) 
                
                err[:]=oset[itt,:]-so[:]
                sum+=np.sum(err[:]**2) # cumul pour calcul de l’erreur rms
                param=wih,who,ni,no,ivec,sh,so,err,deltao,deltah,eta,nh
                
                wih,who,ni,no,ivec,sh,so,err,deltao,deltah,eta,nh=backprop(param) # retropropagation          # retropropagation
    #ENDbouclesurensemble
            
            rmserr[iter]=math.sqrt(sum/nset/no)   # erreur rms a cette iteration
            #ENDbouclesuriterationsd’entrainement
        
            #Maintenantlaphasedetestiraitci-dessous...
            #ENDMAIN
    plt.plot(rmserr,'.',label=str(nh))
   


param=innitialisation()
training(param)


def prediction(param) :
    tset=np.loadtxt('trainingdata.txt',skiprows=1,usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12))
    tset=normalisation(tset) # normalisation
    nset=500 #nombre d'événement tester
    cheatsheet=np.loadtxt('trainingdata.txt',skiprows=1,usecols=(13))[100:100+nset]
    tset=tset[500:500+nset,:]
    reponse=np.zeros(nset)
    for i in range(0,nset) :
        param[4]=tset[i,:]
        param=ffnn(param)
        reponse[i]=param[6]
    return reponse,cheatsheet
 
        
#reponse,cheatsheet=prediction()
#plt.plot(np.abs(cheatsheet-reponse),'.')