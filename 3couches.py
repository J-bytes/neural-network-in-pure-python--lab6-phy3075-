# -*- coding: utf-8 -*-
"""
Created on Sat May  2 18:08:27 2020

@author: joeda
"""


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
   
   
    ni,nh,nn,no,nf,nl    =13,12,4,4,2,1        # nombre d’unites d’entree, interne et de sortie
    wih   =np.zeros([ni,nh])   # poids des connexions entree vers interne
    bih   =np.zeros([ni,nh])
    whn   =np.zeros([nh,nn])   # poids des connexions interne vers sortie
    bhn   =np.zeros([nh,nn])
    wno   =np.zeros([nn,no])   # poids des connexions interne vers sortie
    bno   =np.zeros([nn,no])
    wof   =np.zeros([no,nf])
    bof   =np.zeros([no,nf])
    wfl   =np.zeros([nf,nl])
    bfl   =np.zeros([nf,nl])
    ivec  =np.zeros(ni)        # signal en entree
    sh    =np.zeros(nh)        # signal des neurones interne
    sn    =np.zeros(nn)        # signal des neurones interne
    so    =np.zeros(no)        # signal des neurones de sortie
    sf    =np.zeros(nf)
    sl    =np.zeros(nl)
    err   =np.zeros(nl)        # signal d’erreur des neurones de sortie
    deltal=np.zeros(nl)
    deltaf=np.zeros(nf)
    deltao=np.zeros(no)        # gradient d’erreur des neurones de sortie
    deltan=np.zeros(nn)        # gradient d’erreur des neurones internes
    deltah=np.zeros(nh)        # gradient d’erreur des neurones internes
    eta   =0.1                 # parametre d’apprentissage
  
    
    wih   =np.random.uniform(-0.5,0.5,[ni,nh])   # poids des connexions entree vers interne
    whn   =np.random.uniform(-0.5,0.5,[nh,nn])   # poids des connexions interne vers sortie
    wno   =np.random.uniform(-0.5,0.5,[nn,no])   # poids des connexions interne vers sortie
    wof   =np.random.uniform(-0.5,0.5,[no,nf])
    wfl   =np.random.uniform(-0.5,0.5,[nf,nl])
    #bih   =np.random.uniform(-0.5,0.5,[ni,nh])
    #bhn   =np.random.uniform(-0.5,0.5,[nh,nn])
    #bno   =np.random.uniform(-0.5,0.5,[nn,no])
    #bof   =np.random.uniform(-0.5,0.5,[no,nf])
    #bfl   =np.random.uniform(-0.5,0.5,[nf,nl])
    param=(wih,whn,wno,wof,wfl,ni,nh,nn,no,nf,nl,ivec,sh,so,sn,sf,sl,err,deltao,deltah,deltan,deltaf,deltal,eta,bih,bhn,bno,bof,bfl)
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
    wih,whn,wno,wof,wfl,ni,nh,nn,no,nf,nl,ivec,sh,so,sn,sf,sl,err,deltao,deltah,deltan,deltaf,deltal,eta,bih,bhn,bno,bof,bfl=param

    sh[:]=actv( np.sum(wih*ivec.reshape(ni,1)+bih,axis=0))        #
    # couche interne a couche de sortie
    shtemp=sh.reshape(nh,1)    
    # Eq. (6.3) avec b_1=0
    sn[:]=actv(np.sum(whn[:,:]*shtemp[:,:]+bhn,axis=0))              # Eq. (6.4)
    
    sntemp=sn.reshape(nn,1)
    # Eq. (6.3) avec b_1=0
    so[:]=actv(np.sum(wno[:,:]*sntemp[:,:]+bno,axis=0))              # Eq. (6.4)
    
      
    sotemp=so.reshape(no,1)
    # Eq. (6.3) avec b_1=0
    sf[:]=actv(np.sum(wof[:,:]*sotemp[:,:]+bof,axis=0)) 
    
      
      
    sftemp=sf.reshape(nf,1)
    # Eq. (6.3) avec b_1=0
    sl[:]=actv(np.sum(wfl[:,:]*sftemp[:,:]+bfl,axis=0)) 
    return wih,whn,wno,wof,wfl,ni,nh,nn,no,nf,nl,ivec,sh,so,sn,sf,sl,err,deltao,deltah,deltan,deltaf,deltal,eta,bih,bhn,bno,bof,bfl
#ENDfonctionffnn
#----------------------------------------------------------------------
#retropropagation du signal d’erreur et ajustement des poids du reseau
def backprop(param):
    wih,whn,wno,wof,wfl,ni,nh,nn,no,nf,nl,ivec,sh,so,sn,sf,sl,err,deltao,deltah,deltan,deltaf,deltal,eta,bih,bhn,bno,bof,bfl=param
    """
    deltao[:]=err[:]* dactv(so[:])      # Eq. (6.20)
        
    who[:,:]+=eta*deltao[:]*np.array([sh[:],]*no).transpose()   # Eq. (6.17) pour les wHOpour les wHO
    """
    #for io in range(0,no): # couche de sortie a couche interne
    deltal[:]=err[:]* dactv(sl[:]) # Eq. (6.20)
  
        
    wfl[:,:]+=eta*np.outer(deltal[:],sf[:]).transpose() # Eq. (6.17) pour les wHO
    bfl[:,:]+=eta*deltal[:]
    sum=np.sum((deltal[:]*wfl[:,:]),axis=1)
    deltaf[:]=dactv(sf[:])*sum           # Eq. (6.21)
    
      
    wof[:,:]+=eta*np.outer(deltaf[:],so[:]).transpose() # Eq. (6.17) pour les wHO
    bof[:,:]+=eta*deltaf[:]
    sum=np.sum((deltaf[:]*wof[:,:]),axis=1)
    deltao[:]=dactv(so[:])*sum           # Eq. (6.21)
    
    wno[:,:]+=eta*np.outer(deltao[:],sn[:]).transpose() # Eq. (6.17) pour les wHO
    bno[:,:]+=eta*deltao[:]
    sum=np.sum((deltao[:]*wno[:,:]),axis=1)
    deltan[:]=dactv(sn[:])*sum           # Eq. (6.21)
        
    whn[:,:]+=eta*np.outer(deltan[:],sh[:]).transpose() # Eq. (6.17) pour les wIH pour les wIH
    bhn[:,:]+=eta*deltan[:]
    sum=np.sum((deltan[:]*whn[:,:]),axis=1)
    deltah[:]=dactv(sh[:])*sum           # Eq. (6.21)
        
    wih[:,:]+=eta*np.outer(deltah[:],ivec[:]).transpose() # Eq. (6.17) pour les wIH pour les wIH
    bih[:,:]+=eta*deltah[:]
    return wih,whn,wno,wof,wfl,ni,nh,nn,no,nf,nl,ivec,sh,so,sn,sf,sl,err,deltao,deltah,deltan,deltaf,deltal,eta,bih,bhn,bno,bof,bfl
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
    wih,whn,wno,wof,wfl,ni,nh,nn,no,nf,nl,ivec,sh,so,sn,sf,sl,err,deltao,deltah,deltan,deltaf,deltal,eta,bih,bhn,bno,bof,bfl=param
    nset  =1200              # nombre de membres dans ensemble d’entrainement
    niter =1000                # nombre d’iterations d’entrainement
    oset  =np.zeros([nset,nl]) # sortie pour l’ensemble d’entrainement
    tset  =np.zeros([nset,ni]) # vecteurs-entree l’ensemble d’entrainement
    rmserr=np.zeros(niter)     # erreur rms d’entrainement
    
    #lecture/initialisationdel’ensembled’entrainement
    
    tset=np.loadtxt('trainingdata.txt',skiprows=1,usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12))[0:nset,:]                          # diverses instructions...
    oset[:,0]=np.loadtxt('trainingdata.txt',skiprows=1,usecols=(13))[0:nset]  
    tset=normalisation(tset)
    
          
     # Event ID inutile???                             # diverses instructions...
    #initialisationaleatoiredespoids
   
    
    for iter in range(0,niter):       # boucle sur les iteration d’entrainement
            sum=0.
            print(iter)
            rvec=randomize(nset)           # melange des membres
            for itrain in range(0,nset):   # boucle sur l’ensemble d’entrainement
                itt=rvec[itrain]            # le membre choisi...
                ivec=tset[itt,:]            # ...et son vecteur d’entree
                param=wih,whn,wno,wof,wfl,ni,nh,nn,no,nf,nl,ivec,sh,so,sn,sf,sl,err,deltao,deltah,deltan,deltaf,deltal,eta,bih,bhn,bno,bof,bfl
                wih,whn,wno,wof,wfl,ni,nh,nn,no,nf,nl,ivec,sh,so,sn,sf,sl,err,deltao,deltah,deltan,deltaf,deltal,eta,bih,bhn,bno,bof,bfl=ffnn(param) 
              
                err[:]=oset[itt,:]-sl[:]
                sum+=np.sum(err[:]**2) # cumul pour calcul de l’erreur rms
                param= wih,whn,wno,wof,wfl,ni,nh,nn,no,nf,nl,ivec,sh,so,sn,sf,sl,err,deltao,deltah,deltan,deltaf,deltal,eta,bih,bhn,bno,bof,bfl
                
                wih,whn,wno,wof,wfl,ni,nh,nn,no,nf,nl,ivec,sh,so,sn,sf,sl,err,deltao,deltah,deltan,deltaf,deltal,eta,bih,bhn,bno,bof,bfl=backprop(param) # retropropagation          # retropropagation
    #ENDbouclesurensemble
            
            rmserr[iter]=math.sqrt(sum/nset/no)   # erreur rms a cette iteration
            #ENDbouclesuriterationsd’entrainement
        
            #Maintenantlaphasedetestiraitci-dessous...
            #ENDMAIN
    plt.plot(rmserr,'.',label=str(nh))
    plt.semilogx()
    plt.show()
    return wih,whn,wno,wof,wfl,ni,nh,nn,no,nf,nl,ivec,sh,so,sn,sf,sl,err,deltao,deltah,deltan,deltaf,deltal,eta,bih,bhn,bno,bof,bfl
   

def prediction(param) :
    wih,whn,wno,wof,wfl,ni,nh,nn,no,nf,nl,ivec,sh,so,sn,sf,sl,err,deltao,deltah,deltan,deltaf,deltal,eta,bih,bhn,bno,bof,bfl=param
    tset=np.loadtxt('trainingdata.txt',skiprows=1,usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12))
    tset=normalisation(tset) # normalisation
    nset=1000 #nombre d'événement tester
    r=np.random.randint(0,2000,size=nset)
    tset=tset[r]
    cheatsheet=np.loadtxt('trainingdata.txt',skiprows=1,usecols=(13))[r]
    reponse=np.zeros(nset)
    for i in range(0,nset) :
        ivec=tset[i,:]
        wih,whn,wno,wof,wfl,ni,nh,nn,no,nf,nl,ivec,sh,so,sn,sf,sl,err,deltao,deltah,deltan,deltaf,deltal,eta,bih,bhn,bno,bof,bfl=ffnn([wih,whn,wno,wof,wfl,ni,nh,nn,no,nf,nl,ivec,sh,so,sn,sf,sl,err,deltao,deltah,deltan,deltaf,deltal,eta,bih,bhn,bno,bof,bfl])
        reponse[i]=sl
    return reponse,cheatsheet

param=innitialisation()
param=training(param)
reponse,cheatsheet=prediction(param)
plt.plot(np.abs(cheatsheet-reponse),'.')
plt.show()
print(np.abs(cheatsheet-reponse).mean())


 
        
#reponse,cheatsheet=prediction()
#plt.plot(np.abs(cheatsheet-reponse),'.')