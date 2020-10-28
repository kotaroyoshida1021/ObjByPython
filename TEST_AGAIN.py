import numpy as np
from numba import jit
from scipy.integrate import odeint
from scipy import integrate
from scipy import optimize
from Coordinates import Coordinates
from Ritz import *
import math
import matplotlib.pyplot as plt

length = 1.076991
NDIV = 501
ds = length/(NDIV-1)
NCOORD_PER_FUNC = 8
NCOORD = 3*NCOORD_PER_FUNC + 5
forUfunc = Ritz(NCOORD_PER_FUNC,length)
forOmgXifunc = Ritz(NCOORD_PER_FUNC,length)
forOmgEtafunc = Ritz(NCOORD_PER_FUNC,length)
forInitObj = np.zeros(5)

coef = np.array([[forUfunc.vector],[forOmgXifunc.vector],[forOmgEtafunc.vector],[forInitObj]])
Optvec = coef.ravel()
obj_L = Coordinates(omegaXiL,omegaEtaL,omegaZetaL,length,NDIV)

def omegaXiL(s):
    return 0.0
def omegaEtaL(s):
    return 2.917010
def omegaZetaL(s):
    return 0.0

def initialize():
    chi = 0.0
    delta = 0.343004-math.pi/2.0 
    Schi = math.sin(chi);
    Cchi = math.cos(chi);
    Sdelta = math.sin(delta);
    Cdelta = math.cos(delta);

    xi0 = np.array([Cdelta,0,-Sdelta])
    eta0 = np.array([Schi*Sdelta,Cchi,Schi*Cdelta])
    zeta0 = np.array([Cchi*Sdelta,-Schi,Cchi*Cdelta])
    obj_L.DetermineAxies(xi0,eta0,zeta0)

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

OMG_XI = np.zeros(NDIV)
ALPHA = np.zeros(NDIV)
LAMBDA = np.zeros(NDIV)
Dis = np.zeros(NDIV)
UFUNC_SDOT = np.zeros(NDIV)
UFUNC = np.zeros(NDIV)
Ubend_Sdot = np.zeros(NDIV)
Cond_Sdot = np.zeros(NDIV)


obj_U = Coordinates(omegaXiU,omegaEtaU,omegaZetaU,length,NDIV)

def UpdateDatas():
    for i in range[NDIV]:
        UFUNC_SDOT[i] = UfuncSdot(i*ds)
    #about object_coordinates: 
    UfuncForIntegral = LinearFunc(UFUNC_SDOT,NDIV)
    TMP = np.array([forInitObj[0],forInitObj[1],forInitObj[2]])
    Xi0 = TMP/np.linalg.norm(TMP)
    TMP = np.array([forInitObj[3],forInitObj[4],(-Xi0[0]*forInitObj[3]-Xi0[1]*forInitObj[4])/Xi0[2]])
    Eta0 = TMP/np.linalg.norm(TMP)
    Zeta0 = np.outer(Xi0,Eta0)
    obj_U.DetermineAxies(xi0,Eta0,Zeta0)
    def renewZeta_x(s):
        return UfuncForIntegral.func(s)*obj_U.zeta(s)[0]
    def renewZeta_y(s):
        return UfuncForIntegral.func(s)*obj_U.zeta(s)[1]
    def renewZeta_z(s):
        return UfuncForIntegral.func(s)*obj_U.zeta(s)[2]
    DS = np.linspace(0,length,NDIV)
    for i in range[NDIV]:
        UFUNC[i] = integrate.quad(LinearFunc.func,0.0,i*ds)
        obj_U.pos_x[i] = integrate.quad(renewZeta_x,0.0,i*ds)
        obj_U.pos_y[i] = integrate.quad(renewZeta_y,0.0,i*ds)
        obj_U.pos_z[i] = integrate.quad(renewZeta_z,0.0,i*ds)

    #calculate variables
    for i in range[NDIV]:
        diff = obj_U.pos(i*ds) - obj_L.pos(i*ds)
        Dis[i] = np.linalg.norm(diff)
        if Dis[i]==0.0:
            d_2 = np.zeros(3)
        else:
            d_2 = diff/Dis[i]
        ALPHA[i] = -math.asin(d_2.dot(obj_L.zeta(i*ds)))
        
        tmp = obj_L.zeta(s).outer(obj_U.zeta(s))
        eta = tmp/np.linalg.norm(tmp)
        xi = eta.outer(obj_L.zeta(s))
        OMG_XI[i] = -obj_L.zetaSdot(s).dot(eta)
        LAMBDA[i] = obj_L.zetaSdot(s).dot(xi)
        
    #for integral
    alphaSdot = np.gradient(ALPHA,ds)
    for i in range[NDIV]:
        Calpha = math.cos(ALPHA[i])
        D = Dis[i]
        tmpa = alphaSdot[i]+LMD[i]
        if D>=Calpha/tmpa:
            W = 1.0e+10
        else:
            W = 0.0
        tmp = Calpha/(Calpha-D*tmpa)
        tmp2 = ((OMG_XI[i]/Calpha)**2)/tmpa
        if abs(tmp)<=1.0e-5:
            Ubend_Sdot[i] = 5.0*fabs(tmp2*(1.0-tmp))
        else:
            Ubend_Sdot[i] = 5.0*tmp2*math.log(fabs(tmp))

        ET = obj_L.zeta(s).outer(obj_U.zeta(u))
        diff = obj_U.pos(i*ds)-obj_L.pos(i*ds)
        Cond_Sdot = abs(ET.dot(diff))

   

        
    
    
        

def objective(s):
    return s**2


#main
initialize()
optimize.fmin(objective,1)

plt.plot(TES.pos_z,TES.pos_x)
plt.show()

