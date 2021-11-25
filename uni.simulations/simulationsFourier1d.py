import myUtilities

import numpy as np
import matplotlib.pyplot as plt

import time

### parameters ###
SCRIPTNAME = "simulationFourier"
OVERWRITEOUTPUT = True

L = 4*2*np.pi
n = 32*64
numberOfTimestepsPerUnitMin = 1
T_0 = 0
T_end = 1
### /parameters ###

def myFFT(function):
    #return np.fft.fft(function)/np.sqrt(n)
    return np.fft.fft(function,n)
def myIFFT(functionHat):
    #return np.fft.ifft(functionHat)*np.sqrt(n)
    return np.fft.ifft(functionHat,n)
    #return np.fft.ifft(functionHat)

myUtilities.init(SCRIPTNAME, OVERWRITEOUTPUT)

nHighPrecision = n*1024

xHighPrecision = np.linspace(0,L,nHighPrecision)
freqHighPrecision = np.fft.fftfreq(nHighPrecision)
x = xHighPrecision[:n]
freq = freqHighPrecision[:n]
omega = freq[:n]*nHighPrecision

icHighPrecision = np.sin(xHighPrecision)
#icHat = np.zeros(n)
#icHat[1] = 1
#icHat[0:10]=[0,1,2,1,2,3,4,1,2,3]
#icHat[11:20]=[0,1,2,4,1,2,1,2,3]

icHatHighPrecision = np.fft.fft(icHighPrecision,nHighPrecision)
icHat = icHatHighPrecision[:n]


ic = myIFFT(icHat)
uHat = myFFT(ic)
icHat = myFFT(ic)
print("l2(icHat) "+str(sum(icHat**2)))
print("l2(ic) "+str(sum(abs(ic**2))))
print("l2(uHat) "+str(sum(abs(uHat))))

u = myIFFT(uHat)

print("fft")

plt.subplot(321)
plt.plot(x,ic)
plt.subplot(322)
plt.plot(abs(icHat[:(n/2)]))

plt.subplot(323)
plt.plot(x,u)
plt.subplot(324)
plt.plot(abs(uHat[:(n/2)]))

omega[128:]=0

t=T_0
t_i = 0
lastLogTime = time.time()
while t<T_end:
    if t_i < 1000:
        deltaT = 1.0/(10000000.0*max(abs((omega[:]*omega[:]*omega[:]*omega[:]-omega[:]*omega[:])*uHat[:])))
    else:
        deltaT = 0.01
    uHat[:] = uHat[:]-deltaT*((omega[:]*omega[:]*omega[:]*omega[:]-omega[:]*omega[:])*uHat[:])
    if (time.time()-lastLogTime)>1:
        print(str(np.round(t/(T_end-T_0)*100,2))+"%")
    t += deltaT
    t_i += 1


#print(uHat)

#for 

plt.subplot(325)
plt.plot(x,myIFFT(uHat))

plt.subplot(326)
plt.plot(abs(uHat[:(n/2)]))
plt.show()








myUtilities.writeLog()
