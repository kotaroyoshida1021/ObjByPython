import numpy as np
import numba
import scipy.integrate
from scipy import integrate
from scipy.integrate import odeint
from scipy.linalg import block_diag

class Coordinates(object):
    def __init__(self,omgXi,omgEta,omgZeta,length,MAX):
        self.NDIV = MAX
        self.length = length
        self.omegaXi = omgXi
        self.omegaEta = omgEta
        self.omegaZeta = omgZeta
        self.Ds = self.length/float(self.NDIV-1);
        
        #print('Coordinates emerge\n')

    def Sdot(self,X,s):
        Y = np.reshape(X,[3,3]).copy();
        OmgMat = np.array([[0.0,self.omegaZeta(s),-self.omegaEta(s)],[-self.omegaZeta(s),0.0,self.omegaXi(s)],[self.omegaEta(s),-self.omegaXi(s),0.0]])
        dXds = np.dot(OmgMat,Y)
        return dXds.flatten();
        return dXds

    def DetermineAxies(self,xi0,eta0,zeta0):
        self.pos_x = np.zeros(self.NDIV)
        self.pos_y = np.zeros(self.NDIV)
        self.pos_z = np.zeros(self.NDIV)
        tmp = np.array([xi0,eta0,zeta0])
        X0 = tmp.flatten()
        #print(X0)
        S = np.linspace(0.0,self.length,self.NDIV)
        
        self.X1 = odeint(self.Sdot,X0,S)
        for i in range(self.NDIV):
            RAN = float(i)*self.Ds
            if i==0:
               self.pos_x[i] = 0.0
               self.pos_y[i] = 0.0
               self.pos_z[i] = 0.0
            else:
               self.pos_x[i],_ = integrate.quad(self.__zeta_x,0.0,RAN)
               self.pos_y[i],_ = integrate.quad(self.__zeta_y,0.0,RAN)
               self.pos_z[i],_ = integrate.quad(self.__zeta_z,0.0,RAN)
        TMP= np.array([self.X1[:,6],self.X1[:,7],self.X1[:,8]])
        self.ZETA = TMP.T 
    def renewPos():
        for i in range(self.NDIV):
            RAN = float(i)*self.Ds
            if i==0:
               self.pos_x[i] = 0.0
               self.pos_y[i] = 0.0
               self.pos_z[i] = 0.0
            else:
               self.pos_x[i],_ = integrate.quad(self.__zeta_x,0.0,RAN)
               self.pos_y[i],_ = integrate.quad(self.__zeta_y,0.0,RAN)
               self.pos_z[i],_ = integrate.quad(self.__zeta_z,0.0,RAN)
            #pass
            
         
        #self.X = np.reshape(X1,[-1,3,3]);
    def XI(self,i):
        ret = np.array([self.X1[i,0],self.X1[i,1],self.X1[i,2]])
        return ret
    def ETA(self,i):
        ret = np.array([self.X1[i,3],self.X1[i,4],self.X1[i,5]])
        return ret
    def zeta(self,s):
        p = s/self.Ds;
        n = int(p);
        q = p - float(n);
        if q == 0:
            ret = np.array([self.X1[n,6],self.X1[n,7],self.X1[n,8]])
            return ret
        else :
            ret_n = np.array([self.X1[n,6],self.X1[n,7],self.X1[n,8]])
            ret_n1 = np.array([self.X1[n+1,6],self.X1[n+1,7],self.X1[n+1,8]])
            return (1.0-q)*ret_n + q*ret_n1
    def xi(self,s):
        
        p = s/self.Ds;
        n = int(p);
        q = p - float(n);
        if q == 0:
            ret = np.array([self.X1[n,0],self.X1[n,1],self.X1[n,2]])
            return ret
        else :
            ret_n = np.array([self.X1[n,0],self.X1[n,1],self.X1[n,2]])
            ret_n1 = np.array([self.X1[n+1,0],self.X1[n+1,1],self.X1[n+1,2]])
            return (1.0-q)*ret_n + q*ret_n1

    def eta(self,s):
        
        p = s/self.Ds;
        n = int(p);
        q = p - float(n);
        if q == 0:
            ret = np.array([self.X1[n,3],self.X1[n,4],self.X1[n,5]])
            return ret
        else :
            ret_n = np.array([self.X1[n,3],self.X1[n,4],self.X1[n,5]])
            ret_n1 = np.array([self.X1[n+1,3],self.X1[n+1,4],self.X1[n+1,5]])
            return (1.0-q)*ret_n + q*ret_n1

    def zetaSdot(self,s):
        return self.omegaEta(s)*self.xi(s) - self.omegaXi(s)*self.eta(s)
    def ZETA_SDOT(self,i):
        return self.XI(i)*self.omegaEta(i*self.Ds) - self.ETA(i)*self.omegaXi(i*self.Ds)
    def xiSdot(self,s):
        return self.omegaZeta(s)*self.eta(s) - self.omegaEta(s)*self.zeta(s)
    def etaSdot(self,s):
        return self.omegaXi(s)*self.zeta(s) - self.omegaZeta(s)*self.xi(s)
    def __zeta_x(self,s):
        p = s/self.Ds;
        n = int(p);
        q = p - float(n);
        if q == 0:
            ret = float(self.X1[n,6])
            return ret
        else :
            ret_n = float(self.X1[n,6])
            ret_n1 = float(self.X1[n,6])
            return (1.0-q)*ret_n + q*ret_n1

    def __zeta_y(self,s):
        p = s/self.Ds;
        n = int(p);
        q = p - float(n);
        if q == 0:
            ret = float(self.X1[n,7])
            return ret
        else :
            ret_n = float(self.X1[n,7])
            ret_n1 = float(self.X1[n,7])
            return (1.0-q)*ret_n + q*ret_n1

    def __zeta_z(self,s):
        p = s/self.Ds;
        n = int(p);
        q = p - float(n);
        if q == 0:
            ret = float(self.X1[n,8])
            return ret
        else :
            ret_n = float(self.X1[n,8])
            ret_n1 = float(self.X1[n,8])
            return (1.0-q)*ret_n + q*ret_n1
    def pos(self,s):
        p = s/self.Ds
        n = int(p)
        if(n>self.NDIV):
            print("error! s = ",s)
        q = p - float(n)
        tmp_x = self.pos_x[n]
        tmp_y = self.pos_y[n]
        tmp_z = self.pos_z[n]
        if q == 0:
            return np.array([tmp_x,tmp_y,tmp_z])
        else:
            x_n = self.pos_x[n+1]*q + tmp_x*(1.0-q)
            y_n = self.pos_y[n+1]*q + tmp_y*(1.0-q) 
            z_n = self.pos_z[n+1]*q + tmp_z*(1.0-q)
            return np.array([x_n,y_n,z_n])

     
    """description of class"""


