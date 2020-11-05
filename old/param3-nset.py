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
def setup() :
    global wih,who,ni,no,ivec,sh,so,err,deltao,deltah,eta,nh
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
def ffnn(ivec,nh):
    for ih in range(0,nh):           # couche d’entree a couche interne
        sh[ih]=actv( np.sum(wih[:,ih]*ivec[:]))        # Eq. (6.1)        # Eq. (6.2)
    # couche interne a couche de sortie
    shtemp=np.array([sh,]*no).transpose()    
    # Eq. (6.3) avec b_1=0
    so[:]=actv(np.sum(who[:,:]*shtemp[:,:],axis=0))              # Eq. (6.4)
    return 
#ENDfonctionffnn
#----------------------------------------------------------------------
#retropropagation du signal d’erreur et ajustement des poids du reseau
def backprop(err,nh):
    """
    deltao[:]=err[:]* dactv(so[:])      # Eq. (6.20)
        
    who[:,:]+=eta*deltao[:]*np.array([sh[:],]*no).transpose()   # Eq. (6.17) pour les wHOpour les wHO
    """
    for io in range(0,no): # couche de sortie a couche interne
        deltao[io]=err[io]* dactv(so[io]) # Eq. (6.20)
        for ih in range(0,nh):
            who[ih,io]+=eta*deltao[io]*sh[ih] # Eq. (6.17) pour les wHO
    for ih in range(0,nh):           # couche interne a couche de sortie
        
        sum=np.sum(deltao[:]*who[ih,:])
        deltah[ih]=dactv(sh[ih])*sum           # Eq. (6.21)
        
        wih[:,ih]+=eta*deltah[ih]*ivec[:] # Eq. (6.17) pour les wIH pour les wIH
    return
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
def training(nset) :
    global wih,who,ni,no
    #nset  =200                 # nombre de membres dans ensemble d’entrainement
    niter =1000                # nombre d’iterations d’entrainement
    oset  =np.zeros([nset,no]) # sortie pour l’ensemble d’entrainement
    tset  =np.zeros([nset,ni]) # vecteurs-entree l’ensemble d’entrainement
    rmserr=np.zeros(niter)     # erreur rms d’entrainement
    
    #lecture/initialisationdel’ensembled’entrainement
    
    tset=np.loadtxt('trainingdata.txt',skiprows=1,usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12))[0:nset,:]                          # diverses instructions...
    oset[:,0]=np.loadtxt('trainingdata.txt',skiprows=1,usecols=(13))[0:nset]  
    for i in range(0,13) :
        tset[:,i]/=np.max(tset[:,i]) # normalisation
    
          
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
                ffnn(ivec,nh) 
               
                err[:]=oset[itt,:]-so[:]
                sum+=np.sum(err[:]**2) # cumul pour calcul de l’erreur rms
                backprop(err,nh) # retropropagation          # retropropagation
    #ENDbouclesurensemble
            
            rmserr[iter]=math.sqrt(sum/nset/no)   # erreur rms a cette iteration
            #ENDbouclesuriterationsd’entrainement
        
            #Maintenantlaphasedetestiraitci-dessous...
            #ENDMAIN
    plt.plot(rmserr,'.',label='nset='+str(nset))
   

for nset in [100,200,500,800,1000] :
    setup()
    training(nset)

plt.legend()
plt.xlabel('iteration')
plt.ylabel('Erreurs')

def prediction() :
    tset=np.loadtxt('trainingdata.txt',skiprows=1,usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12))
    for i in range(0,13) :
        tset[:,i]/=np.max(tset[:,i]) # normalisation
    nset=50 #nombre d'événement tester
    cheatsheet=np.loadtxt('trainingdata.txt',skiprows=1,usecols=(13))[100:100+nset]
    tset=tset[100:100+nset,:]
    
    for i in range(0,nset) :
        ffnn(tset[i,:])
    return so,cheatsheet
 
        
#reponse,cheatsheet=prediction()
#plt.plot(np.abs(cheatsheet-reponse),'.')


