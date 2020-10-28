import numpy as np
from scipy.integrate import odeint
from scipy import integrate
from scipy import optimize
from Coordinates import Coordinates
from Ritz import *
from numba import jit
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax
from Multiplier import Multiplier
from time import sleep
import sys
def ProjectedVector(a,b):
    return a - a.dot(b)*b
def normalize(v):
    l_2 = np.linalg.norm(v)
    if l_2==0:
        l_2=1
    return v/l_2
#length = 1.076991
global length,NDIV,ds,NCOORD_PER_FUNC,NCOORD,forUfunc,forOmgXifunc,forOmgEtafunc,InitTheta,InitPhi,InitPsi,obj_L,ncond,nineq

length = math.pi
NDIV = 501
ds = length/(NDIV-1)
NCOORD_PER_FUNC = 8
NCOORD = 3*NCOORD_PER_FUNC + 3
forUfunc = Ritz(NCOORD_PER_FUNC,length)
forOmgXifunc = Ritz(NCOORD_PER_FUNC,length)
forOmgEtafunc = Ritz(NCOORD_PER_FUNC,length)
InitTheta = 0.0 
InitPhi = 0.0 
InitPsi = 0.0
ncond = 4
nineq = 0
#TargetPosition = []

def omegaXiL(s):
    return 0.0
def omegaEtaL(s):
    #return 2.917010
    return 1.0
def omegaZetaL(s):
    return 0.0

obj_L = Coordinates(omegaXiL,omegaEtaL,omegaZetaL,length,NDIV)

def dividedVector(a):
    #forUfunc.vector = a[0:NCOORD_PER_FUNC]
    #forOmgXifunc.vector = a[NCOORD_PER_FUNC:2*NCOORD_PER_FUNC]
    #forOmgEtafunc.vector = a[2*NCOORD_PER_FUNC:3*NCOORD_PER_FUNC]
    for i in range(NCOORD_PER_FUNC):
        forUfunc.vector[i] = a[i]
        forOmgXifunc.vector[i] = a[i+NCOORD_PER_FUNC]
        forOmgEtafunc.vector[i] = a[i+2*NCOORD_PER_FUNC]
    InitPhi = a[3*NCOORD_PER_FUNC]
    InitTheta = a[3*NCOORD_PER_FUNC+1]
    InitPsi = a[NCOORD-1]

def initialize():
    chi = 0.0
    #delta = 0.343004-math.pi/2.0 
    delta = -math.pi/2.0
    Schi = math.sin(chi);
    Cchi = math.cos(chi);
    Sdelta = math.sin(delta);
    Cdelta = math.cos(delta);

    xi0 = np.array([Cdelta,0,-Sdelta])
    eta0 = np.array([Schi*Sdelta,Cchi,Schi*Cdelta])
    zeta0 = np.array([Cchi*Sdelta,-Schi,Cchi*Cdelta])
    forUfunc.vector = np.zeros(NCOORD_PER_FUNC)
    forOmgXifunc.vector = np.zeros(NCOORD_PER_FUNC)
    forOmgEtafunc.vector = np.zeros(NCOORD_PER_FUNC)
    InitPhi=0
    InitPsi=0
    InitTheta=0
    obj_L.DetermineAxies(xi0,eta0,zeta0)
    global TargetPosition,MAX
    TargetPosition= np.loadtxt("./data.txt",comments="#")
    MAX = TargetPosition.shape[0]

def UfuncSdot(s):
    return forUfunc.Func(s)**2
def Ufunc(s):
    return integrate.quad(UfuncSdot,0.0,s)
def omegaXiU(s):
    return forOmgXifunc.Func(s)*UfuncSdot(s)
def omegaEtaU(s):
    return forOmgEtafunc.Func(s)*UfuncSdot(s)
def omegaZetaU(s):
    return 0.0

global OMG_XI,ALPHA,D_COND,ALPHASDOT,XI,ETA,LAMBDA,Dis,UFUNC_SDOT,UFUNC,gene,obj_U
gene = np.empty((NDIV,3))
UFUNC_SDOT = np.zeros(NDIV)
UFUNC = np.zeros(NDIV)
ALPHA = np.zeros(NDIV)
XI = np.zeros((NDIV,3))
ETA = np.zeros((NDIV,3))
LAMBDA = np.zeros(NDIV)
OMG_XI = np.zeros(NDIV)
D_COND = np.zeros(NDIV)
ALPHASDOT = np.empty(NDIV)
obj_U = Coordinates(omegaXiU,omegaEtaU,omegaZetaU,length,NDIV)
#@jit
def UpdateDatas():
    for i in range(NDIV):
        UFUNC_SDOT[i] = UfuncSdot(i*ds)
    #about object_coordinates: 
    UfuncForIntegral = LinearFunc(UFUNC_SDOT,length)
    Cphi = math.cos(InitPhi) 
    Sphi = math.sin(InitPhi)
    Ctheta = math.cos(InitTheta)
    Stheta = math.sin(InitTheta)
    Cpsi = math.cos(InitPsi)
    Spsi = math.sin(InitPsi)
    Xi0 = np.array([Ctheta*Cphi*Cpsi-Sphi*Spsi,Ctheta*Sphi*Cpsi+Cphi*Spsi,-Stheta*Cpsi])
    Eta0 = np.array([-Ctheta*Cphi*Spsi-Sphi*Cpsi,-Ctheta*Sphi*Spsi+Cphi*Cpsi,Stheta*Spsi])
    Zeta0 = np.array([Stheta*Cphi,Stheta*Sphi,Ctheta])
    obj_U.DetermineAxies(Xi0,Eta0,Zeta0)
    def renewZeta_x(s):
        return UfuncForIntegral.func(s)*obj_U.zeta(s)[0]
    def renewZeta_y(s):
        return UfuncForIntegral.func(s)*obj_U.zeta(s)[1]
    def renewZeta_z(s):
        return UfuncForIntegral.func(s)*obj_U.zeta(s)[2]

    for i in range(NDIV):
        UFUNC[i],_ = integrate.quad(UfuncForIntegral.func,0.0,i*ds)
        obj_U.pos_x[i],_ = integrate.quad(renewZeta_x,0.0,i*ds)
        obj_U.pos_y[i],_ = integrate.quad(renewZeta_y,0.0,i*ds)
        obj_U.pos_z[i],_ = integrate.quad(renewZeta_z,0.0,i*ds)
    I = range(NDIV)
    diff_x = obj_U.pos_x - obj_L.pos_x
    diff_y = obj_U.pos_y - obj_L.pos_y
    diff_z = obj_U.pos_z - obj_L.pos_z
    TMP = np.array([diff_x,diff_y,diff_z])
    diff = TMP.T
    #print(diff)
    #print(diff.shape)
    for i in range(NDIV):
        #D = np.linalg.norm(diff[i])
        #D = math.sqrt(np.dot(diff[i],diff[i]))
        TMP = np.array([diff_x[i],diff_y[i],diff_z[i]])
        D = np.linalg.norm(TMP)
        if D==0:
            gene[i] = np.zeros(3)
        else:
            #gene[i] = diff[i]/D
            gene[i] = TMP/D
   
    #print(type(obj_L.ZETA))
    #ALPHA = -math.asin(obj_L.ZETA[I].dot(gene[I]))
    for i in range(NDIV):
        Z = obj_L.ZETA[i]
        G = gene[i]
        A = -math.asin(Z.dot(G))
        ALPHA[i] = A
        X = G/math.cos(A) + math.tan(A)*Z
        XI[i] = X
        N = np.cross(Z,X)
        ETA[i] = N
        ZSD = obj_L.ZETA_SDOT(i)
        OMG_XI = -np.dot(ZSD,X)
        LAMBDA = np.dot(ZSD,N)
    ALPHASDOT = np.gradient(ALPHA)


def setLinearFuncs():
    global Alpha,OmgXi,Lambda,Dist,AlphaSdot,generatrix,xiVec,etaVec
    Alpha = LinearFunc(ALPHA,length)
    OmgXi = LinearFunc(OMG_XI,length)
    Lambda = LinearFunc(LAMBDA,length)
   # Dist = LinearFunc(Dis,length)
    AlphaSdot = LinearFunc(np.array(ALPHASDOT),length)
    generatrix = VectorLinearFunc(gene,length)
    xiVec = VectorLinearFunc(XI,length)
    etaVec = VectorLinearFunc(ETA,length)

#@jit
'''
    この関数は，初期値位置処理回避のため，関数をオーバーラッピングするものである．
    同じsignならばnewtonCG,違うならbrent
'''
def thetaSdot(s,i):
    pass
def objective(a):
    #print(a)
    dividedVector(a)
    UpdateDatas()
    setLinearFuncs()
    ret = 0.0
   
    def thetaInfo(s,i):
        pos_T = np.array([TargetPosition[i,2],TargetPosition[i,0],TargetPosition[i,1]])
        pos_L = obj_L.pos(s)
        G = generatrix.func(s)
        t = G.dot(pos_T - pos_L)
        Qvec = pos_T - pos_L - t*G
        tmp = np.dot(Qvec,etaVec.func(s))/(np.linalg.norm(Qvec)*np.linalg.norm(etaVec.func(s)))
        THT = math.acos(abs(tmp))
        A = Alpha.func(s)
        d_1 = obj_L.zeta(s)*math.cos(A) + xiVec.func(s)*math.sin(A)
        return np.sign(np.dot(Qvec,d_1))*THT

    for i in range(MAX):
        #print("counter = %d \r"%(i))
        def signed_theta(s):
            #print(thetaInfo(s,i))
            return thetaInfo(s,i)
        if np.sign(signed_theta(0)) == np.sign(signed_theta(length)):
            s_i = length
            print("same sign i = %d"%i)
        else:
            s_i = optimize.brentq(signed_theta,0,length)
        pos_T = np.array([TargetPosition[i,2],TargetPosition[i,0],TargetPosition[i,1]])
        pos_L = obj_L.pos(s_i)
        G = generatrix.func(s_i)
        t = G.dot(pos_T - pos_L)
        Qvec = pos_T - pos_L - t*G
        ret += np.linalg.norm(Qvec)
        
    return ret

def calc_conds(a,COND):
    dividedVector(a)
    UpdateDatas()
    setLinearFuncs()
    S = np.linspace(0,length,NDIV)
    C_1 = obj_U.pos(length)-obj_L.pos(length)
    C_2 = integrate.simps(D_COND,S)
    COND = np.array([C_1,C_2])
 
def calc_ineq(a,INEQ):
    pass
count = 0
def cbf(a):
    #ax.set_xlabel("x", size = 14)
    #ax.set_ylabel("y", size = 14)
    #ax.set_zlabel("z", size = 14)
    #ax.plot(obj_L.pos_z,obj_L.pos_x,obj_L.pos_y,color="red")
    #ax.plot(obj_U.pos_z,obj_U.pos_x,obj_U.pos_y,color="blue")
    #fig = plt.figure()
    #ax = fig.gca(projection = '3d')
    #ax.plot(TargetPosition[:,0],TargetPosition[:,1],TargetPosition[:,2],color="green")
    #ax.plot(obj_L.pos_z,obj_L.pos_x,obj_L.pos_y,color="red")
    #ax.plot(obj_U.pos_z,obj_U.pos_x,obj_U.pos_y,color="blue")
    #ax.legend()
    #plt.pause(0.6)
    #plt.show()
    #sleep(3)
    #plt.close()
    #pass
    global count
    
    if count%100 ==0:
        print(a)
    count += 1
    f = objective(a)
    print("\r count :%d    f = %f"%(count,f))
    #
    #    #plt.scatter(count,f)
   
    

def main():
    initialize()
    init = np.ones(NCOORD)*0.001
    plt.ion()
    Multi = Multiplier(objective,init,'nelder-mead',calc_conds,calc_ineq,NCOORD,ncond,nineq,cbf)
    eps = 1.6e-7
    OptimizedResult = Multi.Launch(eps)
    cbf(OptimizedResult.x)

main()
#main


