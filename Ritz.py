import numpy as np
from numba.decorators import jit
import math
import matplotlib.pyplot as plt

BMAX = 10000

class Ritz(object):
    def __init__(self,NCOORD,length):
        self.NCOORD = NCOORD
        self.length = length
        self.q = math.pi/self.length
        self.BASE_FUNCS = [step,ramp,sin1,cos1,sin2,cos2,sin3,cos3,sin4,cos4,sin5,cos5,sin6,cos6,sin7,cos7,sin8,cos8,sin9,cos9,sin10,cos10,sin11,cos11,sin12,cos12,sin13,cos13]
        #self.baseFunc = np.zeros((BMAX,self.NCOORD))
        self.baseFunc = []
        for i in range(self.NCOORD):
            self.baseFunc.append(self.BASE_FUNCS[i])
        
        self.ds = length/float(BMAX-1)
        self.vector = np.zeros(self.NCOORD)
        
        #for i in range(BMAX):
         #   for j in range(self.NCOORD):
          #      self.baseFunc[i,j] = self.BASE_FUNCS[j](self.q,i*self.ds)
    def Func(self,s):
        #p = s/self.ds;
        #n = int(p);
        #q = p - float(n);
        #if n>BMAX:
        #    print("Error")
        #    print(n,s)
        #if q == 0:
        #    tmp = self.baseFunc[n,]
        #    return self.vector.dot(tmp)
        #else:
         #   tmp_n = self.baseFunc[n,]
          #  tmp_n1 = self.baseFunc[n+1,]
           # return (1.0-q)*self.vector.dot(tmp_n) + q*self.vector.dot(tmp_n1)
        tmp = []
        for i in range(self.NCOORD):
            tmp.append(self.baseFunc[i](self.q,s))
        base = np.array(tmp)
        return base.dot(self.vector)
class LinearFunc(object):
    def __init__(self,ARRAY,L_MAX):
        self.FUNC = ARRAY
        self.NDIV = ARRAY.size
        self.length = L_MAX
        self.ds = self.length/(self.NDIV-1)
    def func(self,s):
        p = s/self.ds;
        n = int(p);
        q = p - float(n);
        if q == 0:
            return self.FUNC[n]
        else:
            return (q)*self.FUNC[n+1] + (1-q)*self.FUNC[n]
        
class VectorLinearFunc(object):
    def __init__(self, ARRAY,L_MAX):
        self.FUNC = ARRAY
        self.NDIV = ARRAY.shape[0]
        self.length = L_MAX
        self.ds = self.length/(self.NDIV-1)
    def func(self,s):
        p = s/self.ds;
        n = int(p);
        q = p - float(n);
        if q==0:
            return self.FUNC[n]
        else:
            return (1.0-q)*self.FUNC[n+1] + q*self.FUNC[n]






def step(q,s):
    return 1.0
def ramp(q,s):
    return q*s/math.pi
def sin1(q,s):
    return math.sin(1*q*s)
def sin2(q,s):
    return math.sin(2*q*s)
def sin3(q,s):
    return math.sin(3*q*s)
def sin4(q,s):
    return math.sin(4*q*s)
def sin5(q,s):
    return math.sin(5*q*s)
def sin6(q,s):
    return math.sin(6*q*s)
def sin7(q,s):
    return math.sin(7*q*s)
def sin8(q,s):
    return math.sin(8*q*s)
def sin9(q,s):
    return math.sin(9*q*s)
def sin10(q,s):
    return math.sin(10*q*s)
def sin11(q,s):
    return math.sin(11*q*s)
def sin12(q,s):
    return math.sin(12*q*s)
def sin13(q,s):
    return math.sin(13*q*s)
def sin14(q,s):
    return math.sin(14*q*s)
def sin15(q,s):
    return math.sin(15*q*s)


def cos1(q,s):
    return math.cos(1*q*s)
def cos2(q,s):
    return math.cos(2*q*s)
def cos3(q,s):
    return math.cos(3*q*s)
def cos4(q,s):
    return math.cos(4*q*s)
def cos5(q,s):
    return math.cos(5*q*s)
def cos6(q,s):
    return math.cos(6*q*s)
def cos7(q,s):
    return math.cos(7*q*s)
def cos8(q,s):
    return math.cos(8*q*s)
def cos9(q,s):
    return math.cos(9*q*s)
def cos10(q,s):
    return math.cos(10*q*s)
def cos11(q,s):
    return math.cos(11*q*s)
def cos12(q,s):
    return math.cos(12*q*s)
def cos13(q,s):
    return math.cos(13*q*s)
def cos14(q,s):
    return math.cos(14*q*s)
def cos15(q,s):
    return math.cos(15*q*s)