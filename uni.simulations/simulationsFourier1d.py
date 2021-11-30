import myUtilities

import numpy as np
import matplotlib.pyplot as plt

import time
import datetime
from numpy import zeros
from sympy.matrices import inverse
from sympy.plotting.pygletplot.plot_object import PlotObject

### parameters ###
SCRIPTNAME = "simulationFourier"
OVERWRITEOUTPUT = True

# pde names: heat, burgers, viscBurgers, kuraSiva, kdv, kskdv
pdeName = "kskdv"
# ic names:    temp, sin, box, random
icName = "random"
L = 128
nPerL = 1
deltaT = 0.01/np.power(nPerL,4)         # max seems to be (for nPerL>=1) deltaT = 0.02/np.power(nPerL,4)
rampUpToDeltaT = False                  # start width deltaT small and then increase deltaT by a factor each iteration
restrictRHSsmallerUhat = False           # decrease deltaT such that deltaT*rhs<uHat such that the change per time step in every mode is always smaller then the function itself
restrictRHSsmallerUhatMaxRecursionLength = 10
forceICreal = True 
forceUreal = True
T_0 = 0
T_end = 100000
icScale = 0.00001
plotEveryXticks = 100

plotOnlyAtTheEnd = False

viscBurgersEpsilon = 0.01

# parameters epsilon u_xxx + delta u_xx + gamma u_xxxx
# also in ks, kdv
kappa = 1/16
epsilon = np.sqrt(1-kappa**2)
delta = kappa
gamma = kappa

exportEveryXtimeValues = -1          # not implemented yet negative -> no export
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
 

n = round(L*nPerL)

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
    inverseOrderReal = fHat.real[::-1]
    fHat.real[round(n/2)+1:] = inverseOrderReal[round(n/2):-1]
    inverseOrderImag = fHat.imag[::-1]
    fHat.imag[round(n/2)+1:] = -inverseOrderImag[round(n/2):-1]
    return fHat


myUtilities.init(SCRIPTNAME, OVERWRITEOUTPUT)


x = np.linspace(0,L,n)
omega = myOmega(n,L)


if icName in ["temp"]:
    icHat = zeros(n)
    icHat[0]=0
    icHat[:17]=np.random.rand(17)-0.5
    icHat[16] = icScale
    icHat[15]= icScale
    ic = myIFFT(icHat)
if icName in ["sin"]:
    ic = icScale*np.sin(2*np.pi*x/L)
if icName in ["box"]:
    ic = zeros(n)
    ic[round(n/4):round(3*n/4)]=icScale
if icName in ["random"]:
    ic = icScale*np.random.rand(n)
    ic -= np.sum(ic)/n
     

icHat = myFFT(ic)
if forceICreal:
    icHat = makeHatFourierOfRealFunction(icHat)
    ic = myIFFT(icHat)

uHat = icHat
u = ic



print("l2(icHat)\t"+str(sum(icHat**2)))
print("l2(ic)\t\t"+str(sum(abs(ic**2))))
print("l2(uHat)\t"+str(sum(abs(uHat))))

print("max omega\t"+str(max(abs(omega))))
print("dx\t"+str(round(1/nPerL,3)))


fig = plt.figure()
#ax1 = fig.add_subplot(5,2,1)
#plt.plot(x.real)
#plt.plot(x.imag, color="orange")
#ax2 = fig.add_subplot(5,2,2)
#plt.plot(omega.real)
#plt.plot(omega.imag, color="orange")



axIc = fig.add_subplot(2,2,1)
axIc.title.set_text("ic")
plt.plot(x,ic.real)
plt.plot(x,ic.imag, color="orange")

axIcHat = fig.add_subplot(2,2,2)
axIcHat.title.set_text("ic hat")
plt.plot(omega,icHat.real)
plt.plot(omega,icHat.imag, color="orange")


#plt.plot(xHighPrecision,icHatHighPrecision.real,xHighPrecision,icHatHighPrecision.imag)

axU = plt.subplot(2,2,3)
axU.title.set_text("u (x)")
plt.plot(x,ic.real)
plt.plot(x,ic.imag, color="orange")
axUhat = plt.subplot(2,2,4)
axUhat.title.set_text("u hat (omega)")
plt.plot(omega,icHat.real)
plt.plot(omega,icHat.imag, color="orange")

print(sum(np.power(icHat.real,2)+np.power(icHat.imag,2)))

if not plotOnlyAtTheEnd:
    plt.ion()
    plt.show()
    plt.pause(0.001)

t=T_0
t_i = 0
lastLogTime = time.time()
lastLogTi = 0
lastPlot = 0
lastPlotPercent = 0


exportNumber = 0
if exportEveryXtimeValues>=0:
    numberOfExports = round((T_end-T_0)/exportEveryXtimeValues)
    uRealExport = zeros((numberOfExports,n))
    uImagExport = zeros((numberOfExports,n))
    uHatRealExport = zeros((numberOfExports,n))
    uHatImagExport = zeros((numberOfExports,n))
else:
    numberOfExports = -1


def getRHS(pdeName=pdeName):
    if pdeName in ["heat","heatEquation"]:
        return -(np.power(omega,2))*uHat
    if pdeName in ["burgers"]:
        uX = myIFFT(1j*omega*uHat)
        uuxHat = myFFT(myIFFT(uHat)*uX) 
        return -uuxHat
    if pdeName in ["viscBurgers"]:
        uX = myIFFT(1j*omega*uHat)
        uuxHat = myFFT(myIFFT(uHat)*uX)
        return -(uuxHat + viscBurgersEpsilon*(np.power(omega,2))*uHat)
    if pdeName in ["kuraSivaLin"]:
        return (delta*np.power(omega,2)-gamma*np.power(omega,4))*uHat
    if pdeName in ["kuraSiva"]:
        return getRHS(pdeName="kuraSivaLin")+getRHS(pdeName="burgers")
    if pdeName in ["kdv"]:
        uxxxHat = -1j*np.power(omega,3)*uHat
        return getRHS("burgers")-epsilon*uxxxHat-0.0001*(np.power(omega,4))*uHat
    if pdeName in ["kskdv"]:
        uxxxHat = -1j*np.power(omega,3)*uHat
        return getRHS(pdeName="kuraSiva")-epsilon*uxxxHat
    
    
    
def getConstSTaGeqCb(a,b,c, recursionsLeft=-1):
    if recursionsLeft == 0:
        return c
    if (np.abs(a)>np.abs(c*b)).all():
        #print(c)
        return c
    else:
        return getConstSTaGeqCb(a, b, c/2.0, recursionsLeft-1)
    
simulationStartTime = datetime.datetime.now()
if rampUpToDeltaT:
    deltaTadjusted = 0.00000000001
else:
    deltaTadjusted = deltaT
    
    
while t<T_end:
    if forceUreal:
        uHat = makeHatFourierOfRealFunction(uHat)
    rhs = getRHS()
    if rampUpToDeltaT:
        if deltaTadjusted < deltaT:
            deltaTadjusted *= 1.01
    if restrictRHSsmallerUhat:
        deltaTadjusted = getConstSTaGeqCb(uHat, rhs, deltaT, restrictRHSsmallerUhatMaxRecursionLength)
    uHat = uHat + deltaTadjusted*rhs
    
    if numberOfExports>0:
        if t/(T_end-T_0)>exportNumber/numberOfExports:
            u = myIFFT(uHat)
            uRealExport[exportNumber,:] = u.real
            uImagExport[exportNumber,:] = u.imag
            uHatRealExport[exportNumber,:] = uHat.real
            uHatImagExport[exportNumber,:] = uHat.imag
            exportNumber += 1
    if (time.time()-lastLogTime)>1:
        simulationSpeed = (t_i-lastLogTi)/(time.time()-lastLogTime)
        lastLogTime = time.time()
        now = datetime.datetime.now()
        if t>0.0000000001:
            timeLeft = (now-simulationStartTime)*((T_end-T_0)/t-1)
        else:
            timeLeft = "error"
        print(str(np.round(t/(T_end-T_0)*100,2))+"%, t_i = "+str(t_i)+", t = "+str(round(t,2))+", time left "+str(timeLeft)+", time passed "+str(now-simulationStartTime)+", speed = "+str(round(simulationSpeed))+" t_i/sec"+ ", deltaT = "+str(deltaTadjusted))
        lastLogTi = t_i
    #uHat = (1-myOmega*myOmega)*uHat
    if t_i >= lastPlot + plotEveryXticks:
        lastPlot = t_i
        u = myIFFT(uHat)
        if not plotOnlyAtTheEnd:
            axU.clear()
            axU.title.set_text("u (x)")
            axUhat.clear()
            axUhat.title.set_text("u hat (omega)")
        axU.plot(x,u.real)
        axUhat.plot(omega,uHat.real)
        if not plotOnlyAtTheEnd:
            axU.plot(x,u.imag, color="orange")
            axUhat.plot(omega,uHat.imag, color="orange")
            plt.pause(0.001)
        else:
            axU.plot(x,u.imag)
            axUhat.plot(omega,uHat.imag)   
        
    t += deltaTadjusted
    t_i += 1


u = myIFFT(uHat)

#plt.subplot(5,2,7)
#plt.plot(x,u.real)
#plt.plot(x,u.imag, color="orange")
#plt.subplot(5,2,8)
#plt.plot(omega,uHat.real)
#plt.plot(omega,uHat.imag, color="orange")

#fig2 = plt.figure()
#axRealStateSpace = fig2.add_subplot(2,2,1,projection="3d")
#axImagStateSpace = fig2.add_subplot(2,2,2,projection="3d")
#axRealFourierSpace = fig2.add_subplot(2,2,3,projection="3d")
#axImagFourierSpace = fig2.add_subplot(2,2,4,projection="3d")
#axFourierSpace = fig.add_subplot(5210,projection="3d")

if numberOfExports>0:
    for j in range(uRealExport.shape[0]):
        #todo export
        print(" ")



if plotOnlyAtTheEnd:
    plt.show()


myUtilities.writeLog()

### stop closing the view
input("Press [enter] to continue.")
