import numpy as np
from scipy.integrate import odeint
from scipy import integrate
from scipy import optimize
import sys
MAX = 30
class Multiplier(object):
    def __init__(self,obj,x0,method,calc_cond,calc_ineq,coord_num,ncond,nineq,cbf):
        self.obj = obj
        self.x0 = x0
        self.method = method
        self.calc_cond = calc_cond
        self.calc_ineq = calc_ineq
        self.NCOND = coord_num
        self.ncond = ncond
        self.nineq = nineq
        self.Lambda = np.zeros(self.ncond)
        self.Mu = np.zeros(self.nineq)
        self.r = np.ones(self.ncond)*10
        self.s = np.ones(self.nineq)*10
        self.c = sys.float_info.max
        self.alpha = 10
        self.beta = 0.25
        self.cbf = cbf
    def objective(self,a):
        COND = np.zeros(self.ncond)
        INEQ = np.zeros(self.nineq)
        self.calc_cond(a,COND)
        self.calc_ineq(a,INEQ)
        F_Cond = np.dot(COND,self.Lambda) + np.dot(COND*self.Lambda,COND)
        TMP = self.Mu + self.s*INEQ
        def INEQ_FUNC(i):
            if TMP[i]<0:
                return -0.5*self.Mu[i]*self.Mu[i]/self.s[i]
            else:
                return self.Mu[i]*INEQ[i]+0.5*self.s[i]*INEQ[i]**2
        F_Ineq = 0
        for i in range(self.nineq):
            F_Ineq += INEQ_FUNC(i)
        return self.obj(a) + F_Cond + F_Ineq
    
    
    def Launch(self,epsilon):
        for i in range(MAX):
            print("================",i,"-th iteration================")
            a = optimize.minimize(self.objective,self.x0,method = self.method,callback=self.cbf,options={'xatol':1.0e-8,'fatol':1.0e-12})
            print(a)
            G = np.zeros(self.ncond)
            H = np.zeros(self.nineq)
            COND = np.zeros(self.ncond)
            INEQ = np.zeros(self.nineq)
            self.calc_cond(a.x,COND)
            self.calc_ineq(a.x,INEQ)
            G = abs(COND)
            H = np.maximum(H,-self.Mu/self.s)
            #print(type(H))
            try:
                G_max = np.amax(G)
            except ValueError:
                G_max = 0
            try:
                H_max = np.amax(H)
            except ValueError:
                H_max = -1
            c = max(G_max,H_max)
            if c<epsilon:
                break
            else:
                self.Lambda = self.Lambda + self.r*COND
                Z = np.zeros(self.nineq)
                self.Mu = np.maximum(Z,self.Mu+self.s*INEQ)
                for i in range(self.ncond):
                    if G[i]<self.beta*c:
                        self.r[i] *= self.alpha
                for i in range(self.nineq):
                    if H[i]<self.beta*c:
                        self.s[i] *= self.alpha
        return a
            
    """description of class"""


