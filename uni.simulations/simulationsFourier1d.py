import myUtilities

import numpy as np
import matplotlib.pyplot as plt

import time
import datetime
from numpy import zeros
from sympy.matrices import inverse

### parameters ###
SCRIPTNAME = "simulationFourier"
OVERWRITEOUTPUT = True

L = 2*np.pi*4
n = 512
deltaT = 1
T_0 = 0
T_end = 10
### /parameters ###

### todo
    ### todo omega² oder omega⁴ * small schaukelt sich auf
    ### sin(x) für L=4pi passt nicht
    ### fourier transform amplitude checken



### FFT guide ###
# only for np.fft.fft, np.fft.ifft, np.fft.fftfreq
# not for np.fft.rfft, np.fft.irfft or np.fft.hfft, np.fft.ihfft

## general
# the (inverse) fft produces not the fourier transform but rather the "e^(i 2 pi freq)-fourier transform"
# so fft of real 3 lowest modes sin function is given by 0 9 27 29  ... ... 29 27 9
# as the freq given by fftfreq is given by 0 1 2 3 ... ... -3 -2 -1
# so for the real 3 modes sin function the pos and negative freq in e^(i 2 pi freq) have to match to cancel the imaginary parts
# highest mode is therefore in the middle (for even n) array position -> icHat[n/2]=1 yields zickzack

## freq scale
# if fHat = fft(f) then fHat[1] is always the longest wave that fits in an size n (len of array f) array
# for freq = fftfreq(n) without width, then freq goes from 0 to <0.5 jumps to >=-0.5 and goes to <0
# so it always matches the omega for the circle in the e^(i * ...) representation
# if freq = fftfreq(n,d) d is specified then d (the "spacing") freq(n,d) = freq(n)/(d*n) and for d=1/(Ln) freq(n,d) = L*freq(n)
 

def myFFT(function):
    return np.fft.fft(function, norm="ortho")   # ortho scales unitary, i.e. 1/sqrt(n) for both the fft and the ifft
def myIFFT(functionHat):
    return np.fft.ifft(functionHat, norm="ortho")
def myOmega(n,L):
    # in rad
    return 2*np.pi/L*n*np.fft.fftfreq(n)
    #return np.fft.ifft(functionHat)
def makeHatFourierOfRealFunction(fHat):
    # takes fHat and transforms it so that f is real
    inverseOrder = fHat[::-1]
    fHat[round(n/2)+1:] = inverseOrder[round(n/2):-1]
    return fHat

def check(a,b,c):
    # if c*a < b return c
    # else decrease c
    checkArray = c*a-b
    if max(checkArray)<0:
        c=c
    else:
        print(c*b)
        
        print(a)
        c = checkArrayAmplitudesAconstBiggerThenB(a, b, c/10)
    return c
        

myUtilities.init(SCRIPTNAME, OVERWRITEOUTPUT)


x = np.linspace(0,L,n)
omega = myOmega(n,L)


icHat = zeros(n)
#icHat[0]=0
#icHat[:17]=np.random.rand(17)-0.5
#icHat[16] = 1

#icHat[15]=1

#icHat = makeHatFourierOfRealFunction(icHat)
ic = np.sin(x)  
#ic = myIFFT(icHat)
#ic = zeros(n)
#ic[round(n/4):round(3*n/4)]=1
icHat = myFFT(ic)


print("max abs(omega) = "+str(max(abs(omega))))


uHat = icHat
u = ic
print("l2(icHat) "+str(sum(icHat**2)))
print("l2(ic) "+str(sum(abs(ic**2))))
print("l2(uHat) "+str(sum(abs(uHat))))

u = myIFFT(uHat)
print("fft")

fig = plt.figure()
fig.add_subplot(521)
plt.plot(x.real)
plt.plot(x.imag, color="orange")
fig.add_subplot(522)
plt.plot(omega.real)
plt.plot(omega.imag, color="orange")



fig.add_subplot(523)
plt.plot(ic.real)
plt.plot(ic.imag, color="orange")

fig.add_subplot(524)
plt.plot(icHat.real)
plt.plot(icHat.imag, color="orange")


#plt.plot(xHighPrecision,icHatHighPrecision.real,xHighPrecision,icHatHighPrecision.imag)

plt.subplot(525)
plt.plot(x,ic.real)
plt.plot(x,ic.imag, color="orange")
plt.subplot(526)
plt.plot(omega,icHat.real)
plt.plot(omega,icHat.imag, color="orange")


t=T_0
t_i = 0
lastLogTime = time.time()
lastPlotPercent = 0
plotNumber = 0
plots = 10
uRealExport = zeros((plots,n))
uImagExport = zeros((plots,n))
uHatRealExport = zeros((plots,n))
uHatImagExport = zeros((plots,n))
simulationStartTime = datetime.datetime.now()
while t<T_end:
    rhs = (np.power(omega,2))*uHat
    deltaTadjusted = min(deltaT,1/max(np.power(omega,2)))   
    uHat = uHat - deltaTadjusted*rhs
    if (t/(T_end-T_0)>plotNumber/plots):
        lastPlotPercent = t/(T_end-T_0)
        u = myIFFT(uHat)
        uRealExport[plotNumber,:] = u.real
        uImagExport[plotNumber,:] = u.imag
        uHatRealExport[plotNumber,:] = uHat.real
        uHatImagExport[plotNumber,:] = uHat.imag
        plotNumber += 1
    #    uExport[plotNumber-1,:] = myIFFT(uHat)
    #new = uHat + 1
    #uHat = new
    if (time.time()-lastLogTime)>1:
        lastLogTime = time.time()
        now = datetime.datetime.now()
        if t>0.0000000000001:
            timeLeft = (now-simulationStartTime)*((T_end-T_0)/t-1)
        else:
            timeLeft = "error"
        print(str(np.round(t/(T_end-T_0)*100,2))+"%, t_i = "+str(t_i)+", time left "+str(timeLeft))
    
    #uHat = (1-myOmega*myOmega)*uHat
    t += deltaTadjusted
    t_i += 1



u = myIFFT(uHat)

plt.subplot(527)
plt.plot(x,u.real)
plt.plot(x,u.imag, color="orange")
plt.subplot(528)
plt.plot(omega,uHat.real)
plt.plot(omega,uHat.imag, color="orange")

fig2 = plt.figure()
axRealStateSpace = fig2.add_subplot(221,projection="3d")
axImagStateSpace = fig2.add_subplot(222,projection="3d")
axRealFourierSpace = fig2.add_subplot(223,projection="3d")
axImagFourierSpace = fig2.add_subplot(224,projection="3d")
#axFourierSpace = fig.add_subplot(5210,projection="3d")

for j in range(uRealExport.shape[0]):
    ys = j*np.ones(uRealExport.shape[1])
    axRealStateSpace.plot(x,ys,uRealExport[j,:])
    axImagStateSpace.plot(x,ys,uImagExport[j,:])
    axRealFourierSpace.plot(omega,ys,uHatRealExport[j,:])
    axImagFourierSpace.plot(omega,ys,uHatImagExport[j,:])



plt.show()



myUtilities.writeLog()
