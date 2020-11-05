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

global mw1,mw2,mw3,mw4,mw5,mb1,mb2,mb3,mb4,mb5,vw1,vw2,vw3,vw4,vw5,vb1,vb2,vb3,vb4,vb5

mw1,mw2,mw3,mw4,mw5,mb1,mb2,mb3,mb4,mb5=0,0,0,0,0,0,0,0,0,0




vw1,vw2,vw3,vw4,vw5,vb1,vb2,vb3,vb4,vb5=0,0,0,0,0,0,0,0,0,0

beta1,beta2=0.9,0.999
eps=1e-8
def innitialisation() :
   
   
    ni,nh,nn,no,nf    =13,32,16,16,1        # nombre d’unites d’entree, interne et de sortie
    wih   =np.zeros([ni,nh])   # poids des connexions entree vers interne
    bih   =np.zeros([ni,nh])
    whn   =np.zeros([nh,nn])   # poids des connexions interne vers sortie
    bhn   =np.zeros([nh,nn])
    wno   =np.zeros([nn,no])   # poids des connexions interne vers sortie
    bno   =np.zeros([nn,no])
    wof   =np.zeros([no,nf])
    bof   =np.zeros([no,nf])
    ivec  =np.zeros(ni)        # signal en entree
    sh    =np.zeros(nh)        # signal des neurones interne
    sn    =np.zeros(nn)        # signal des neurones interne
    so    =np.zeros(no)        # signal des neurones de sortie
    sf    =np.zeros(nf)
    err   =np.zeros(nf)        # signal d’erreur des neurones de sortie
    deltaf=np.zeros(nf)
    deltao=np.zeros(no)        # gradient d’erreur des neurones de sortie
    deltan=np.zeros(nn)        # gradient d’erreur des neurones internes
    deltah=np.zeros(nh)        # gradient d’erreur des neurones internes
    eta   =0.001                 # parametre d’apprentissage
  
    
    wih   =np.random.uniform(-0.5,0.5,[ni,nh])   # poids des connexions entree vers interne
    whn   =np.random.uniform(-0.5,0.5,[nh,nn])   # poids des connexions interne vers sortie
    wno   =np.random.uniform(-0.5,0.5,[nn,no])   # poids des connexions interne vers sortie
    wof   =np.random.uniform(-0.5,0.5,[no,nf])
    bih   =np.random.uniform(-0.5,0.5,[ni,nh])
    bhn   =np.random.uniform(-0.5,0.5,[nh,nn])
    #bno   =np.random.uniform(-0.5,0.5,[nn,no])
    #bof   =np.random.uniform(-0.5,0.5,[no,nf])
    param=(wih,whn,wno,wof,ni,nh,nn,no,nf,ivec,sh,so,sn,sf,err,deltao,deltah,deltan,deltaf,eta,nh,bih,bhn,bno,bof)
    return param

def adam(m,v,beta1,beta2,eta,dw) :
    m=beta1*m+(1-beta1)*dw
    v=beta2*v+(1-beta2)*dw**2
    mm=m/(1-beta1)
    vv=v/(1-beta2)
    ww=eta/(vv+eps)**.5*mm
    return m,v,ww
def normalisation(data):
    return (data-data.mean(axis=0))/data.std(axis=0)
#----------------------------------------------------------------------
#Fonction d’activation sigmoidale#???
def sigmoide(a):
    return  1./(1.+np.exp(-a))          # Eq. (6.5)
   # return (np.tanh(a))**2
#----------------------------------------------------------------------
#Derivee de la fonction d’activation sigmoidale
def dsigmoide(s):
    return  s*(1.-s)                 # Eq. (6.19)

def relu(a) :
    return  np.where(a>0,a,0)

def drelu(s) :
    return np.where(s>0,1,0)
#----------------------------------------------------------------------
#fonction reseau feed-forward,calcul signal de sortie
def ffnn(param):
    wih,whn,wno,wof,ni,nh,nn,no,nf,ivec,sh,so,sn,sf,err,deltao,deltah,deltan,deltaf,eta,nh,bih,bhn,bno,bof=param

    sh[:]=relu( np.sum(wih*ivec.reshape(ni,1)+bih,axis=0))        #
    # couche interne a couche de sortie
    shtemp=sh.reshape(nh,1)    
    # Eq. (6.3) avec b_1=0
    sn[:]=relu(np.sum(whn[:,:]*shtemp[:,:]+bhn,axis=0))              # Eq. (6.4)
    
    sntemp=sn.reshape(nn,1)
    # Eq. (6.3) avec b_1=0
    so[:]=relu(np.sum(wno[:,:]*sntemp[:,:]+bno,axis=0))              # Eq. (6.4)
    
      
    sotemp=so.reshape(no,1)
    # Eq. (6.3) avec b_1=0
    sf[:]=sigmoide(np.sum(wof[:,:]*sotemp[:,:]+bof,axis=0)) 
    return  wih,whn,wno,wof,ni,nh,nn,no,nf,ivec,sh,so,sn,sf,err,deltao,deltah,deltan,deltaf,eta,nh,bih,bhn,bno,bof
#ENDfonctionffnn
#----------------------------------------------------------------------
#retropropagation du signal d’erreur et ajustement des poids du reseau
def backprop(param):
    global mw1,mw2,mw3,mw4,mw5,mb1,mb2,mb3,mb4,mb5,vw1,vw2,vw3,vw4,vw5,vb1,vb2,vb3,vb4,vb5
    wih,whn,wno,wof,ni,nh,nn,no,nf,ivec,sh,so,sn,sf,err,deltao,deltah,deltan,deltaf,eta,nh,bih,bhn,bno,bof=param
    """
    deltao[:]=err[:]* dactv(so[:])      # Eq. (6.20)
        
    who[:,:]+=eta*deltao[:]*np.array([sh[:],]*no).transpose()   # Eq. (6.17) pour les wHOpour les wHO
    """
    #for io in range(0,no): # couche de sortie a couche interne
    deltaf[:]=err[:]* dsigmoide(sf[:]) # Eq. (6.20)
  
    dw=np.outer(deltaf[:],so[:]).transpose()
    mw1,vw1,dw=adam(mw1,vw1,beta1,beta2,eta,dw)
    wof[:,:]+=dw # Eq. (6.17) pour les wHO

    db=deltaf[:]
    mb1,vb1,db=adam(mb1,vb1,beta1,beta2,eta,db)
    bof[:,:]+=db
    
    
    sum=np.sum((deltaf[:]*wof[:,:]),axis=1)
    deltao[:]=drelu(so[:])*sum           # Eq. (6.21)
    
    dw=np.outer(deltao[:],sn[:]).transpose()
    mw2,vw2,dw=adam(mw2,vw2,beta1,beta2,eta,dw)
   
    wno[:,:]+=dw # Eq. (6.17) pour les wHO
    db=deltao[:]
    mb2,vb2,db=adam(mb2,vb2,beta1,beta2,eta,db)
    bno[:,:]+=db
    
    sum=np.sum((deltao[:]*wno[:,:]),axis=1)
    deltan[:]=drelu(sn[:])*sum           # Eq. (6.21)
    
    dw=np.outer(deltan[:],sh[:]).transpose()
    mw3,vw3,dw=adam(mw3,vw3,beta1,beta2,eta,dw)
    whn[:,:]+=dw # Eq. (6.17) pour les wIH pour les wIH
  
    db=deltan[:]
    mb3,vb3,db=adam(mb3,vb3,beta1,beta2,eta,db)
    bhn[:,:]+=db
    
    
    sum=np.sum((deltan[:]*whn[:,:]),axis=1)
    deltah[:]=drelu(sh[:])*sum           # Eq. (6.21)
        
    dw=np.outer(deltah[:],ivec[:]).transpose()
    mw4,vw4,dw=adam(mw4,vw4,beta1,beta2,eta,dw)
    wih[:,:]+=dw # Eq. (6.17) pour les wIH pour les wIH
    
    db=deltah[:]
    mb4,vb4,db=adam(mb4,vb4,beta1,beta2,eta,db)
    bih[:,:]+=db
    return wih,whn,wno,wof,ni,nh,nn,no,nf,ivec,sh,so,sn,sf,err,deltao,deltah,deltan,deltaf,eta,nh,bih,bhn,bno,bof
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
    wih,whn,wno,wof,ni,nh,nn,no,nf,ivec,sh,so,sn,sf,err,deltao,deltah,deltan,deltaf,eta,nh,bih,bhn,bno,bof=param
    nset  =400              # nombre de membres dans ensemble d’entrainement
    niter =300                # nombre d’iterations d’entrainement
    oset  =np.zeros([nset,nf]) # sortie pour l’ensemble d’entrainement
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
                param=wih,whn,wno,wof,ni,nh,nn,no,nf,ivec,sh,so,sn,sf,err,deltao,deltah,deltan,deltaf,eta,nh,bih,bhn,bno,bof
                wih,whn,wno,wof,ni,nh,nn,no,nf,ivec,sh,so,sn,sf,err,deltao,deltah,deltan,deltaf,eta,nh,bih,bhn,bno,bof=ffnn(param) 
              
                err[:]=oset[itt,:]-sf[:]
                sum+=np.sum(err[:]**2) # cumul pour calcul de l’erreur rms
                param=wih,whn,wno,wof,ni,nh,nn,no,nf,ivec,sh,so,sn,sf,err,deltao,deltah,deltan,deltaf,eta,nh,bih,bhn,bno,bof
                
                wih,whn,wno,wof,ni,nh,nn,no,nf,ivec,sh,so,sn,sf,err,deltao,deltah,deltan,deltaf,eta,nh,bih,bhn,bno,bof=backprop(param) # retropropagation          # retropropagation
    #ENDbouclesurensemble
            
            rmserr[iter]=math.sqrt(sum/nset/nf)   # erreur rms a cette iteration
            #ENDbouclesuriterationsd’entrainement
            if iter%1==0 :
                reponse,cheatsheet=prediction(param)
                plt.plot(iter,len(np.where(np.abs(cheatsheet-np.round(reponse,0))==0)[0])/500,'.',color='red')
            #Maintenantlaphasedetestiraitci-dessous...
            #ENDMAIN
    plt.plot(rmserr,'.',label=str(nh))
    #plt.semilogx()
    plt.show()
    return  wih,whn,wno,wof,ni,nh,nn,no,nf,ivec,sh,so,sn,sf,err,deltao,deltah,deltan,deltaf,eta,nh,bih,bhn,bno,bof
   

def prediction(param) :
    wih,whn,wno,wof,ni,nh,nn,no,nf,ivec,sh,so,sn,sf,err,deltao,deltah,deltan,deltaf,eta,nh,bih,bhn,bno,bof=param
    tset=np.loadtxt('trainingdata.txt',skiprows=1,usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12))
    tset=normalisation(tset) # normalisation
    nset=500 #nombre d'événement tester
  
    tset=tset[1500:2000]
    cheatsheet=np.loadtxt('trainingdata.txt',skiprows=1,usecols=(13))[1500:2000]
    reponse=np.zeros(nset)
    for i in range(0,nset) :
        ivec=tset[i,:]
        wih,whn,wno,wof,ni,nh,nn,no,nf,ivec,sh,so,sn,sf,err,deltao,deltah,deltan,deltaf,eta,nh,bih,bhn,bno,bof=ffnn([wih,whn,wno,wof,ni,nh,nn,no,nf,ivec,sh,so,sn,sf,err,deltao,deltah,deltan,deltaf,eta,nh,bih,bhn,bno,bof])
        reponse[i]=sf
    return reponse,cheatsheet

param=innitialisation()
param=training(param)
reponse,cheatsheet=prediction(param)
plt.plot(np.abs(cheatsheet-reponse),'.')
plt.show()
print(len(np.where(np.abs(cheatsheet-np.round(reponse,0))==0)[0]))


 
        
#reponse,cheatsheet=prediction()
#plt.plot(np.abs(cheatsheet-reponse),'.')