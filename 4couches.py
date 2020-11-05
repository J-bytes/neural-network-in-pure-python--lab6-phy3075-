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
#==================================================================
#Variables globales

global mw1,mw2,mw3,mw4,mw5,mb1,mb2,mb3,mb4,mb5,vw1,vw2,vw3,vw4,vw5,vb1,vb2,vb3,vb4,vb5

mw1,mw2,mw3,mw4,mw5,mb1,mb2,mb3,mb4,mb5=0,0,0,0,0,0,0,0,0,0

vw1,vw2,vw3,vw4,vw5,vb1,vb2,vb3,vb4,vb5=0,0,0,0,0,0,0,0,0,0

global bestf,bestx,m,compteur
compteur=0
bestf=0
beta1,beta2=0.9,0.999
eps=1e-8
m=1

train=pd.read_csv('data/train.csv')


#==================================================================
def innitialisation() :
   
   
    ni,nh,nn,no,nf,nl    =13,20,16,12,8,15        # nombre d’unites d’entree, interne et de sortie
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
    eta   =0.001                 # parametre d’apprentissage
  
    
    #initialisation aléatoire des biais et des poids
    wih   =np.random.uniform(-0.5,0.5,[ni,nh])   
    whn   =np.random.uniform(-0.5,0.5,[nh,nn])  
    wno   =np.random.uniform(-0.5,0.5,[nn,no])  
    wof   =np.random.uniform(-0.5,0.5,[no,nf])
    wfl   =np.random.uniform(-0.5,0.5,[nf,nl])
    bih   =np.random.uniform(-0.5,0.5,[ni,nh])
    bhn   =np.random.uniform(-0.5,0.5,[nh,nn])
    bno   =np.random.uniform(-0.5,0.5,[nn,no])
    bof   =np.random.uniform(-0.5,0.5,[no,nf])
    bfl   =np.random.uniform(-0.5,0.5,[nf,nl])
    param=(wih,whn,wno,wof,wfl,ni,nh,nn,no,nf,nl,ivec,sh,so,sn,sf,sl,err,deltao,deltah,deltan,deltaf,deltal,eta,bih,bhn,bno,bof,bfl)
    return param

#==================================================================
def adam(m,v,beta1,beta2,eta,dw) : 
    """
    

    Parameters
    ----------
    m : variable m au temps t-1.
    v : variable v au temps t-1.
    beta1 : paramètre libre, typiquement 0.9.
    beta2 : paramètre libre, typiquement 0.999.
    eta : paramètre d'apprentissage, typiquement 0.001.
    dw : gradient des poids/biais.

    Returns
    -------
    m : variable m au temps t.
    v :  variable v au temps t.
    ww : variation des poids/biais.

    """
    m=beta1*m+(1-beta1)*dw
    v=beta2*v+(1-beta2)*dw**2
    mm=m/(1-beta1)
    vv=v/(1-beta2)
    ww=eta/(vv+eps)**.5*mm
    return m,v,ww
#==================================================================
def normalisation(data):
    return (data-data.mean(axis=0))/data.std(axis=0) # normalise les données
#----------------------------------------------------------------------
#Fonction d’activation sigmoidale
def sigmoide(a):
    return  1./(1.+np.exp(-a))          # Eq. (6.5)
   # return (np.tanh(a))**2
#----------------------------------------------------------------------
#Derivee de la fonction d’activation sigmoidale
def dsigmoide(s):
    return  s*(1.-s)                 # Eq. (6.19)

#----------------------------------------------------------------------
#fonction d'activation ReLu
def relu(a) :
    return  np.where(a>0,a,0)

#Dérivée de la fonction d'activation ReLu
def drelu(s) :
    return np.where(s>0,1,0)
#----------------------------------------------------------------------
#fonction reseau feed-forward,calcul signal de sortie
def ffnn(param):
    wih,whn,wno,wof,wfl,ni,nh,nn,no,nf,nl,ivec,sh,so,sn,sf,sl,err,deltao,deltah,deltan,deltaf,deltal,eta,bih,bhn,bno,bof,bfl=param

    #Couche d'input à première couche interne
    sh[:]=relu( np.sum(wih*ivec.reshape(ni,1)+bih,axis=0))        
    #-----------------------------
    #couches internes
    
    #1 à 2
    shtemp=sh.reshape(nh,1)    
    sn[:]=relu(np.sum(whn[:,:]*shtemp[:,:]+bhn,axis=0))              
    
    #2 à 3
    sntemp=sn.reshape(nn,1)
    so[:]=sigmoide(np.sum(wno[:,:]*sntemp[:,:]+bno,axis=0))             
    
    #3 à 4
    sotemp=so.reshape(no,1)
    sf[:]=sigmoide(np.sum(wof[:,:]*sotemp[:,:]+bof,axis=0)) 
    
    #------------------------
    #couche interne à couche de sortie 
    sftemp=sf.reshape(nf,1)
    sl[:]=sigmoide(np.sum(wfl[:,:]*sftemp[:,:]+bfl,axis=0)) 
    return wih,whn,wno,wof,wfl,ni,nh,nn,no,nf,nl,ivec,sh,so,sn,sf,sl,err,deltao,deltah,deltan,deltaf,deltal,eta,bih,bhn,bno,bof,bfl
#ENDfonctionffnn
#----------------------------------------------------------------------
#retropropagation du signal d’erreur et ajustement des poids du reseau

def backprop(param):
    """
    Cette fonction permet la rétropropagation de l'erreur RMS pour ajuster les poids grâce 
    à la méthode d'optimisation Adam'

    Parameters
    ----------
    param : Paramètre du réseau neuronal tel que défini dans initialisation.

    Returns : Paramètre du réseau neuronal tel que défini dans initialisation.
    -------
    

    """
    global mw1,mw2,mw3,mw4,mw5,mb1,mb2,mb3,mb4,mb5,vw1,vw2,vw3,vw4,vw5,vb1,vb2,vb3,vb4,vb5
    wih,whn,wno,wof,wfl,ni,nh,nn,no,nf,nl,ivec,sh,so,sn,sf,sl,err,deltao,deltah,deltan,deltaf,deltal,eta,bih,bhn,bno,bof,bfl=param
    
    
    deltal[:]=err[:]* dsigmoide(sl[:]) #évaluation de l'erreur en ce point du réseau
  
    dw=np.outer(deltal[:],sf[:]).transpose() #calcul du gradient 
    mw1,vw1,dw=adam(mw1,vw1,beta1,beta2,eta,dw) #appel d'Adam pour calculer la variation à effectuer sur les poids
   
    wfl[:,:]+=dw #ajustement des poids
    db=deltal[:]
    mb1,vb1,db=adam(mb1,vb1,beta1,beta2,eta,db)#appel d'Adam pour calculer la variation à effectuer sur les biais
   
    bfl[:,:]+=db #ajustement des biais
    
    #L'on répète le processus pour chacune des autres couches (excepté la couche d'input)
    sum=np.sum((deltal[:]*wfl[:,:]),axis=1)
    deltaf[:]=dsigmoide(sf[:])*sum          #évaluation de l'erreur en ce point du réseau
    
    dw=np.outer(deltaf[:],so[:]).transpose() #calcul du gradient 
    mw2,vw2,dw=adam(mw2,vw2,beta1,beta2,eta,dw)#appel d'Adam pour calculer la variation à effectuer sur les poids
   
 
    wof[:,:]+=dw #
    db=deltaf[:]
    mb2,vb2,db=adam(mb2,vb2,beta1,beta2,eta,db)#appel d'Adam pour calculer la variation à effectuer sur les biais
   
    bof[:,:]+=db
    sum=np.sum((deltaf[:]*wof[:,:]),axis=1)
    deltao[:]=dsigmoide(so[:])*sum     #évaluation de l'erreur en ce point du réseau       
    
    dw=np.outer(deltao[:],sn[:]).transpose() #calcul du gradient 
    mw3,vw3,dw=adam(mw3,vw3,beta1,beta2,eta,dw) #appel d'Adam pour calculer la variation à effectuer sur les poids
   
    
    wno[:,:]+=dw 
    
    db=deltao[:]
    mb3,vb3,db=adam(mb3,vb3,beta1,beta2,eta,db) #appel d'Adam pour calculer la variation à effectuer sur les biais
   
    bno[:,:]+=db
    sum=np.sum((deltao[:]*wno[:,:]),axis=1)
    deltan[:]=drelu(sn[:])*sum        #évaluation de l'erreur en ce point du réseau   
        
    dw=np.outer(deltan[:],sh[:]).transpose()  #calcul du gradient 
    mw4,vw4,dw=adam(mw4,vw4,beta1,beta2,eta,dw)  #appel d'Adam pour calculer la variation à effectuer sur les poids
   
    
    
    whn[:,:]+=dw
    
    db=deltan[:]
    mb4,vb4,db=adam(mb4,vb4,beta1,beta2,eta,db) #appel d'Adam pour calculer la variation à effectuer sur les biais
   
    bhn[:,:]+=db
    sum=np.sum((deltan[:]*whn[:,:]),axis=1)
    deltah[:]=drelu(sh[:])*sum           # Eq. (6.21)
        
    dw=np.outer(deltah[:],ivec[:]).transpose()
    mw5,vw5,dw=adam(mw5,vw5,beta1,beta2,eta,dw) #appel d'Adam pour calculer la variation à effectuer sur les poids
   
    
  
    wih[:,:]+=dw # Eq. (6.17) pour les wIH pour les wIH
    
    db=deltah[:]
    mb5,vb5,db=adam(mb5,vb5,beta1,beta2,eta,db) #appel d'Adam pour calculer la variation à effectuer sur les biais
    bih[:,:]+=db
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
    global bestf,bestx,m,compteur
    wih,whn,wno,wof,wfl,ni,nh,nn,no,nf,nl,ivec,sh,so,sn,sf,sl,err,deltao,deltah,deltan,deltaf,deltal,eta,bih,bhn,bno,bof,bfl=param
    nset  =1000              # nombre de membres dans ensemble d’entrainement
    niter =800            # nombre d’iterations d’entrainement
    oset  =np.zeros([nset,nl]) # sortie pour l’ensemble d’entrainement
    tset  =np.zeros([nset,ni]) # vecteurs-entree l’ensemble d’entrainement
    rmserr=np.zeros(niter)     # erreur rms d’entrainement
    
    #lecture/initialisationdel’ensembled’entrainement
    
    tset=np.loadtxt('trainingdata.txt',skiprows=1,usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12))[0:nset,:]                          # diverses instructions...
    oset[:,0]=np.loadtxt('trainingdata.txt',skiprows=1,usecols=(13))[0:nset]  
    
    tset=normalisation(tset)
    
    
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
            
            rmserr[iter]=math.sqrt(sum/nset/1)   # erreur rms a cette iteration
            
            reponse,cheatsheet=prediction(param)
            plt.plot(iter,1-len(np.where(np.abs(cheatsheet-np.round(reponse,0))==0)[0])/1000,'.',color='red')
          
            #Maintenantlaphasedetestiraitci-dessous...
            #ENDMAIN
            err2=np.mean((reponse-cheatsheet)**2)
            rmserr2=math.sqrt(err2)
            plt.plot(iter,rmserr2,'.',color='black')
    plt.plot(rmserr,'.',color='blue')
    #plt.semilogx()
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('Erreur')
    plt.show()
    return wih,whn,wno,wof,wfl,ni,nh,nn,no,nf,nl,ivec,sh,so,sn,sf,sl,err,deltao,deltah,deltan,deltaf,deltal,eta,bih,bhn,bno,bof,bfl
   

def prediction(param) :
    
    #Cette fonction permet de classifier les données non utilisés lors de l'entrainement pour permettre
    #de déterminer quel pourcentage des données sont bien classifier à une itération donnée
    wih,whn,wno,wof,wfl,ni,nh,nn,no,nf,nl,ivec,sh,so,sn,sf,sl,err,deltao,deltah,deltan,deltaf,deltal,eta,bih,bhn,bno,bof,bfl=param
    tset=np.loadtxt('trainingdata.txt',skiprows=1,usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12))
    tset=normalisation(tset) # normalisation
    nset2=1000 #nombre d'événement tester
   
    tset=tset[1000:2000]
    cheatsheet=np.loadtxt('trainingdata.txt',skiprows=1,usecols=(13))[2000-nset2:2000]
    reponse=np.zeros(nset2)
    for i in range(0,nset2) :
        ivec=tset[i,:]
        wih,whn,wno,wof,wfl,ni,nh,nn,no,nf,nl,ivec,sh,so,sn,sf,sl,err,deltao,deltah,deltan,deltaf,deltal,eta,bih,bhn,bno,bof,bfl=ffnn([wih,whn,wno,wof,wfl,ni,nh,nn,no,nf,nl,ivec,sh,so,sn,sf,sl,err,deltao,deltah,deltan,deltaf,deltal,eta,bih,bhn,bno,bof,bfl])
        reponse[i]=sl
    return reponse,cheatsheet


param=innitialisation()
param=training(param)
reponse,cheatsheet=prediction(param)
rep=np.round(reponse,0)
vp=len(np.where(np.logical_and(rep==1,cheatsheet==1))[0])#vrai positif
fp=len(np.where(np.logical_and(rep==1,cheatsheet==0))[0])#faux positif
fn=len(np.where(np.logical_and(rep==0,cheatsheet==1))[0])#faux négatif
vn=len(np.where(np.logical_and(rep==0,cheatsheet==0))[0])#vrai negatif
plt.plot(np.abs(cheatsheet-reponse),'.')
plt.show()
print(len(np.where(np.abs(cheatsheet-np.round(reponse,0))==0)[0]))
#######################################################################
tsetpaul=np.loadtxt('testdata.txt',skiprows=1,usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12))
reponsepaul=np.zeros((len(tsetpaul[:,0])))
wih,whn,wno,wof,wfl,ni,nh,nn,no,nf,nl,ivec,sh,so,sn,sf,sl,err,deltao,deltah,deltan,deltaf,deltal,eta,bih,bhn,bno,bof,bfl=param
for i in range(0,len(tsetpaul[:,0])) :
    ivec=tsetpaul[i,:]
    ivec=normalisation(ivec)
    wih,whn,wno,wof,wfl,ni,nh,nn,no,nf,nl,ivec,sh,so,sn,sf,sl,err,deltao,deltah,deltan,deltaf,deltal,eta,bih,bhn,bno,bof,bfl=ffnn((wih,whn,wno,wof,wfl,ni,nh,nn,no,nf,nl,ivec,sh,so,sn,sf,sl,err,deltao,deltah,deltan,deltaf,deltal,eta,bih,bhn,bno,bof,bfl))
    reponsepaul[i]=np.round(sl,0)
np.savetxt('reponse_paul9.txt',reponsepaul,fmt='%.1d')