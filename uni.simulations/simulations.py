from firedrake import *
import numpy as np
import os

### time ###
import datetime

### code source code file ###
from shutil import copy
from ufl.tensors import as_scalar
from ufl.operators import div, grad
from ufl import indexed
from cmath import sqrt



# using now() to get current time  
time_start = datetime.datetime.now()
print(time_start,"starting ...")  



### PARAMETERS ###


# interval lengths
L = 1
L_x = L
L_y = L

# spatial steps
nProL = 128
n_x = L_x*nProL
n_y = L_y*nProL
#n = 64*2
#n_x = n
#n_y = n

# times
numberOfTimestepsPerUnit = 1000
T_end = 13
timeInitialUadv = 0.01      ### for miles u_adv need a sine flow until t = 0.01 (otherwise get a stationary solution)

# pde name
# list of available shortNames: nonLin, onlyAdv, advLap, advLap2, advLap2Lap, kuraSiva
pdeShortName = "advLap"

# finite elements
finitEleFamily = "CG"           #https://www.firedrakeproject.org/variational-problems.html#supported-finite-elements
finitEleDegree = 1


# force 0 average after every step? True, False12
forceZeroAverage = False

# kappa in theta_t + < u_adv, grad theta> + kappa*laplace theta + laplace^2 theta = 0
kappa = 1/4

### initial condition ###
ic_scale = 1
ic_freq = 1
ic_freq_x = ic_freq
ic_freq_y = ic_freq
randomIC = False

### advection ###
advZero = False
adv_scale = 1
adv_scaleX = adv_scale
adv_scaleY = adv_scale
adv_freq = 1
adv_freqX = adv_freq
adv_freqY = adv_freq


# pde name
# list of available flows: milesEnergy, milesEnstrophy, camillaTestFlow, none (anything else for just initial)
usedFlow = "camillaTestFlow"
### rescale outputs -> \| . \|_2 = 1 ###
rescaleOutputs = True
inverseLaplacianEnforceAverageFreeBefore = True
inverseLaplacianEnforceAverageFreeAfter = True

### write output only every ... time intervals to save storage space
writeOutputEvery = 0.01             # 0 -> every time,
 

### PARAMETERS END ###






T_0 = 0
numberOfTimesteps = round(numberOfTimestepsPerUnit*(T_end-T_0))
timestepInitial = (T_end-T_0)/numberOfTimesteps
timestep = timestepInitial

# mesh = UnitSquareMesh(n, n)
#mesh = PeriodicUnitSquareMesh(n,n)
mesh = PeriodicRectangleMesh(n_x,n_y,L_x,L_y)

V = FunctionSpace(mesh, finitEleFamily, finitEleDegree)
V_vec = VectorFunctionSpace(mesh, finitEleFamily, finitEleDegree)
V_out = FunctionSpace(mesh, finitEleFamily, 1)
x, y = SpatialCoordinate(mesh)


print(datetime.datetime.now(),"spaces defined")  


### initial data ###
ic_theta = Function(V)

ic_theta.interpolate(ic_scale*sin(ic_freq_x*2*pi*x/L_x))
#ic_theta.interpolate(ic_scale*e**(-ic_freq_x*((x-L_x/2)/L_x)**2-ic_freq_y*((y-L_y/2)/L_y)**2))
#ic_theta.interpolate(-scale*cos(freq_x*2*pi*x/L_x))
#ic_theta.interpolate(-scale*cos(freq_y*2*pi*y/L_y))

### random initial data ###
if randomIC:
    u_0Random = ic_scale*(2*np.random.rand(n_x*n_y)-1)       # 2*... - 1 -> -1 und 1
    ic_theta = Function(V,u_0Random)




#u_adv = project(as_vector([sin(adv_freqX*2*pi*y/L_y), sin(adv_freqY*2*pi*x/L_x)]), V_vec)
#u_adv = project(as_vector([adv_scaleX*sin(adv_freqX*2*pi*y/L_y), adv_scaleY*sin(adv_freqY*2*pi*x/L_x)]), V_vec)
u_adv = project(as_vector([adv_scaleX*sin(adv_freqX*2*pi*y/L_y),0]), V_vec)
#u_adv = project(as_vector([y/L_y,0]), V_vec)
#u_adv = project(as_vector([2*(x-L_x/2),-2*(y-L_y/2)]), V_vec)
#u_adv = project(as_vector([1, 1]), V_vec)
if advZero:
    u_adv.assign(0)
    
    
print(datetime.datetime.now(),"inital values assigned ")  





### functions ### 
###################################
def calcInverseLaplacian(function):
    # returns lap^-1 f
    # f=function, u = outFunction
    # lap u = f
    # <lap u, v> =<f,v>
    # <-grad u, grad v> = <f,v>
    # <-grad u, grad v> - <f,v> = 0
    # <grad u, grad v> + <f,v> = 0 
    #print("1",function.dat.data)
    if inverseLaplacianEnforceAverageFreeBefore:
        if norm(getZeroAverageOfScalarFunction(function)-function,"l2")>0.01*((L_x*L_y)**0.5):
            print("!!!warning!!! initial data of get inverse laplacian is non average free -> enforcing average free")
        function = getZeroAverageOfScalarFunction(function)
    
    outFunction = Function(V)
    testFunctionInvLap = TestFunction(V)
    F_invLap = (inner(grad(outFunction),grad(testFunctionInvLap))+dot(function,testFunctionInvLap))*dx
    solve(F_invLap == 0, outFunction)
    if inverseLaplacianEnforceAverageFreeAfter:
        if norm(getZeroAverageOfScalarFunction(outFunction)-outFunction,"l2")>0.01*((L_x*L_y)**0.5):
            print("!!!warning!!! result of get inverse laplacian is non average free -> enforcing average free")
        outFunction = getZeroAverageOfScalarFunction(outFunction)
    return outFunction
###################################
def calcLaplacian(function):
    gradFunction = project(grad(function), V_vec)
    return project(div(gradFunction),V)
###################################
def getZeroAverageOfScalarFunction(function):
    if isinstance(function, Function):
        thisFunction = function
    else:
        thisFunction = project(function,V)      ### when indexed for example theta, theta_laplace = w.split() then theta is not a function but rather Indexed: w_20[0] or something like this
    sum = np.sum(thisFunction.dat.data)
    thisFunction.dat.data[:] = thisFunction.dat.data[:]-sum/(n_x*n_y)
    return thisFunction
###################################
### now miles flow ###
### page 40 https://deepblue.lib.umich.edu/bitstream/handle/2027.42/143927/cmiless_1.pdf?sequence=1
def calcMilesLerayDivFreeProjector(vectorFunction):
    vectorFunctionTemp = project(vectorFunction, V_vec)  ### IF THIS FAILS IT WAS PROBABLY NOT A VECTOR FUNCTION
    diverg = project(div(vectorFunction),V)
    lastTerm = project(grad(calcInverseLaplacian(diverg)), V_vec)
    returnFunction = Function(V_vec)
    returnFunction.dat.data[:] = vectorFunctionTemp.dat.data[:] - lastTerm.dat.data[:]
    return returnFunction
def calcIntegral(function):
    return assemble(function*dx)
def calcSpatialAverage(function):
    return calcIntegral(function)/(L_x*L_y)
def calcAbsOfVectorFunction(function):
    absFunction = Function(V)
    absFunction.dat.data[:] = (function.dat.data[:,0]**2+function.dat.data[:,1]**2)**(1/2)
    return absFunction
def calcMilesOptimalFlowEnergyCase(function):
    funcFunction = project(function, V)
    invLap = calcInverseLaplacian(funcFunction)
    nablaInvLap = project(grad(invLap), V_vec)
    functionNablaLapInvFunction = Function(V_vec)
    functionNablaLapInvFunction.dat.data[:,0] = funcFunction.dat.data[:]*nablaInvLap.dat.data[:,0]
    functionNablaLapInvFunction.dat.data[:,1] = funcFunction.dat.data[:]*nablaInvLap.dat.data[:,1]
    projection = calcMilesLerayDivFreeProjector(functionNablaLapInvFunction)
    #normalisationFactor = 1/((calcSpatialAverage(calcAbsOfVectorFunction(projection))**2)**(1/2))
    normalisationFactor = abs(sqrt(L_x*L_y))/(norm(projection,"l2"))
    #print(normalisationFactor)
    retFunction = projection
    retFunction.dat.data[:] = normalisationFactor*projection.dat.data[:]
    return retFunction
def calcMilesOptimalFlowEnstrophyCase(function):
    funcFunction = project(function, V)
    invLap = calcInverseLaplacian(funcFunction)
    nablaInvLap = project(grad(invLap), V_vec)
    functionNablaLapInvFunction = Function(V_vec)
    functionNablaLapInvFunction.dat.data[:,0] = funcFunction.dat.data[:]*nablaInvLap.dat.data[:,0]
    functionNablaLapInvFunction.dat.data[:,1] = funcFunction.dat.data[:]*nablaInvLap.dat.data[:,1]
    projection = calcMilesLerayDivFreeProjector(functionNablaLapInvFunction)
    projectionX = Function(V)
    projectionY = Function(V)
    projectionX.dat.data[:] = projection.dat.data[:,0]
    projectionY.dat.data[:] = projection.dat.data[:,1]
    minNablaInvunctionNablaLapInvFunction = Function(V_vec)
    minNablaInvunctionNablaLapInvFunction.dat.data[:,0] = -calcInverseLaplacian(projectionX).dat.data[:]
    minNablaInvunctionNablaLapInvFunction.dat.data[:,1] = -calcInverseLaplacian(projectionY).dat.data[:]
    normalisationFactor = abs(sqrt(L_x*L_y))/(norm(grad(minNablaInvunctionNablaLapInvFunction),"l2"))            ### in the code of miles they somehow use curl instead of grad
    #normalisationFactor = 1/(norm((minNablaInvunctionNablaLapInvFunction)))
    #print(normalisationFactor)
    retFunction = minNablaInvunctionNablaLapInvFunction
    retFunction.dat.data[:] = normalisationFactor*minNablaInvunctionNablaLapInvFunction.dat.data[:]
    return retFunction
UinMiles = 0
GammaInMiles = 0
tauInMiles = 0
l_bat = 0
l_dom = 0
l_root = 0
def calcConstants_batscaleGammaUmiles():
    ### recalculates the constants (batchelor scale, Gamma, U(in miles), tau, dominant wavelength for lap2lap 
    global UinMiles
    global GammaInMiles
    global tauInMiles
    global l_bat
    global l_dom
    global l_root
    
    # miles p 5
    UinMiles = abs(norm(u_adv,"l2")/(sqrt(L_x*L_y)))
    GammaInMiles = abs(norm(grad(u_adv),"l2")/(sqrt(L_x*L_y)))
    # irgendwie erkennt es sqrt variables immer als komplexe zahlen mit im part 0 an ...
    
    # miles p 16
    tauInMiles = 1/GammaInMiles
    # miles p 34
    if usedFlow in ['milesEnergy']:
        # miles: 
        l_bat = 3/2*kappa/UinMiles
    elif usedFlow in ['milesEnstrophy']:
        l_bat = abs(sqrt(3/2*kappa*tauInMiles))
    else:
        ### default energy but to see that it is strange put it negativ
        l_bat = -(3/2*kappa/UinMiles)
    
    if pdeShortName in ['advLap2Lap', 'kuraSiva']:
        # irgendwie erkennt es sqrt variables immer als komplexe zahlen mit im part 0 an ...
        l_dom = abs(2*sqrt(2)*pi)
        l_root = abs(2*pi)
    else:
        ### doesn't make sense but still compute and put - to indicate something strange
        l_dom = - abs(2*sqrt(2)*pi)
        l_root = - abs(2*pi)
        
        
def calcCdotOfVectors(f,g):
    result = Function(V)
    result.dat.data[:] = f.dat.data[:,0]*g.dat.data[:,0]+f.dat.data[:,1]*g.dat.data[:,1]
    return result
def calcCamillaTestFlow(function):
    #u_adv = theta nalba^-1 theta - nabla^-1(nabla theta cdot nabla^-1 theta) - nabla^-1 (theta^2)
    nablaMinusOneFunction = calcDivMinus1ofScalar(function)
    GradFunction = project(grad(function),V_vec)
    functionSq = Function(V)
    functionSq.dat.data[:] = function.dat.data[:]**2
    
    firstTerm = Function(V_vec)
    firstTerm.dat.data[:,0] = function.dat.data[:]*nablaMinusOneFunction.dat.data[:,0]
    firstTerm.dat.data[:,1] = function.dat.data[:]*nablaMinusOneFunction.dat.data[:,1]
    
    secondTerm = calcDivMinus1ofScalar(calcCdotOfVectors(GradFunction, nablaMinusOneFunction)) 
    
    thirdTerm = calcDivMinus1ofScalar(functionSq)
    
    result = Function(V_vec)
    result.dat.data[:] = firstTerm.dat.data[:]-secondTerm.dat.data[:]-thirdTerm.dat.data[:]
    result.dat.data[:] = result.dat.data[:]/norm(result,"l2") 
    return result 
    
def calcDivMinus1ofScalar(function):
    # div^{-1} = div^{-1} divgrad laplace^{-1} = grad laplace^{-1}
    return project(grad(calcInverseLaplacian(function)),V_vec)
def calcHminus1NormOfScalar(function):
    return norm(calcDivMinus1ofScalar(function),"l2")
def getComponentOfVectorFunction(function, component):
    retFunction = Function(V)
    retFunction.assign(0)
    retFunction.dat.data[:] = function.dat.data[:,component]
    return retFunction
def getOutputMeshFunctionScalar(function, name, value = None, component = -1):
    if value != None:
        thisFunction = Function(V)
        thisFunction.assign(value)
    else:
        if isinstance(function, Function):
            thisFunction = function
        else:
            thisFunction = project(function,V)      ### when indexed for example theta, theta_laplace = w.split() then theta is not a function but rather Indexed: w_20[0] or something like this
    retFunction = Function(V_out, name = name)
    if component in[0,1]:
        retFunction.dat.data[:] = thisFunction.dat.data[:,component]
    else:
        retFunction.dat.data[:] = thisFunction.dat.data[:]
    if rescaleOutputs:
        if value == None:
            retFunction /= norm(thisFunction,"l2")
    return retFunction
def writeOutputMeshFunctions():
    
    outTheta1 = getOutputMeshFunctionScalar(theta, "theta")
    #outTheta2 = getOutputMeshFunctionScalar(theta, "theta (2)")
    #outTheta3 = getOutputMeshFunctionScalar(theta, "theta (3)")
    
    gradTheta = project(grad(theta), V_vec)
    outGradThetaX = getOutputMeshFunctionScalar(gradTheta, "d/dx theta", None, 0)
    outGradThetaY = getOutputMeshFunctionScalar(gradTheta, "d/dy theta", None, 1)
    
    outUadvX = getOutputMeshFunctionScalar(u_adv, "u_adv x", None, 0)
    outUadvY = getOutputMeshFunctionScalar(u_adv, "u_adv y", None, 1)
    
    #l_domFunction = Function(V).assign(l_dom)
    #outLdom = getOutputMeshFunctionScalar(None,"l_dominant", l_dom)

    #outfile_theta.write(outTheta1, outTheta2, outTheta3, outGradThetaX, outGradThetaY, outUadvX, outUadvY, outLdom, time=t)
    outfile_theta.write(outTheta1, outGradThetaX, outGradThetaY, outUadvX, outUadvY, time=t)





    
    


if pdeShortName in ['nonLin']:
    numberTestFunctions = 1
elif pdeShortName in ['onlyAdv']:
    numberTestFunctions = 1
elif pdeShortName in ['advLap']:
    numberTestFunctions = 1
elif pdeShortName in ['advLap2Lap']:
    numberTestFunctions = 2
elif pdeShortName in ['advLap2']:
    numberTestFunctions = 2
elif pdeShortName in ['kuraSiva']:
    numberTestFunctions = 2
else:
    print("ERROR wrong pde short name (not in list)")


### define the functions
if numberTestFunctions == 1:
    theta = Function(V)
    theta_old = Function(V)
    
    theta.assign(ic_theta)
    theta_old.assign(ic_theta)

    theta_laplace = Function(V)
    theta_laplace_old = Function(V)
    testFunctionA = TestFunction(V)

else:
    W = MixedFunctionSpace((V, V))
    W_out = MixedFunctionSpace((V_out, V_out))
    w_old = Function(W)
    theta_old, theta_laplace_old = w_old.split()
    
    ### not sure about initial condition for laplace theta = 0 since it would have to be laplace ic_theta
    ### but should (and looks like does) work via the pde 
    ic_theta_laplace = 0
    
    theta_old.assign(ic_theta)
    theta_laplace_old.assign(ic_theta_laplace)
    
    w = Function(W)
    w.assign(w_old)
    
    theta, theta_laplace = split(w)
    theta_old, theta_laplace_old = split(w_old)
    
    
    
    testFunctionA, testFunctionB = TestFunctions(W)
    

#################################


#################################



##### pde #####

F_nonLin = (inner((theta - theta_old)/timestep, testFunctionA)
    + 1/2*inner(dot(grad(theta),grad(theta)), testFunctionA) 
    )*dx
    
F_onlyAdv = (inner((theta - theta_old)/timestep, testFunctionA)
    + inner(dot(u_adv,grad(theta)), testFunctionA)
    )*dx

F_advLap = (inner((theta - theta_old)/timestep, testFunctionA)
    + inner(dot(u_adv,grad(theta)), testFunctionA) 
    + kappa * inner(grad(theta), grad(testFunctionA))
    )*dx

if numberTestFunctions == 2:
    F_advLap2Lap = (inner((theta - theta_old)/timestep, testFunctionA)
        + inner(dot(u_adv,grad(theta)), testFunctionA) 
        + kappa*inner(theta_laplace, testFunctionA)
        - kappa*inner(grad(theta_laplace), grad(testFunctionA))
        + inner(theta_laplace, testFunctionB)
        + inner(grad(theta), grad(testFunctionB))
        )*dx
    
    F_advLap2 = (inner((theta - theta_old)/timestep, testFunctionA)
        + inner(dot(u_adv,grad(theta)), testFunctionA)
        - kappa*inner(grad(theta_laplace), grad(testFunctionA))
        + inner(theta_laplace, testFunctionB)
        + inner(grad(theta), grad(testFunctionB))
        )*dx
        
    F_kuraSiva = (inner((theta - theta_old)/timestep, testFunctionA)
        + 1/2*inner(dot(grad(theta),grad(theta)), testFunctionA) 
        + kappa*inner(theta_laplace, testFunctionA)
        - kappa*inner(grad(theta_laplace), grad(testFunctionA))
        + inner(theta_laplace, testFunctionB)
        + inner(grad(theta), grad(testFunctionB))
        )*dx

if pdeShortName in ['nonLin']:
    F = F_nonLin
elif pdeShortName in ['onlyAdv']:
    F = F_onlyAdv
elif pdeShortName in ['advLap']:
    F = F_advLap
elif pdeShortName in ['advLap2Lap']:
    F = F_advLap2Lap
elif pdeShortName in ['advLap2']:
    F = F_advLap2
elif pdeShortName in ['kuraSiva']:
    F = F_kuraSiva
else:
    print("ERROR wrong pde short name (not in list)")


# problem = NonlinearVariationalProblem(F, w, bcs=[bc_u, bc_v])
if numberTestFunctions == 1:
    problem = NonlinearVariationalProblem(F, theta)
else:
    problem = NonlinearVariationalProblem(F, w)
    



### solver ###

# sp_it = {
#    "ksp_type": "gmres",
#    "pc_type": "fieldsplit",
#    "pc_fieldsplit_type": "schur",
#    "pc_fieldsplit_0_fields": "1",
#    "pc_fieldsplit_1_fields": "0",
#    "pc_fieldsplit_schur_precondition": "selfp",
#    "fieldsplit_0_pc_type": "ilu",
#    "fieldsplit_0_ksp_type": "preonly",
#    "fieldsplit_1_ksp_type": "preonly",
#    "fieldsplit_1_pc_type": "gamg",
#    "ksp_monitor": None,
#    "ksp_max_it": 20,
#    "snes_monitor": None
#    }

sp_it = {
    'mat_type': 'aij',
    'ksp_type': 'preonly',
    'pc_type': 'lu'
    }

solver = NonlinearVariationalSolver(problem, solver_parameters=sp_it)


    
print(datetime.datetime.now(),"variational problem defined")  








##### outputs #####
output_dir_path = os.path.dirname(os.path.realpath(__file__))


### output mesh functions ####

t = T_0





outfile_theta = File(output_dir_path + "/../data/temp/theta.pvd")
#outfile_theta.write(outTheta1, outTheta2, outTheta3, outGradThetaX, outGradThetaY, outUadvX, outUadvY, outLdom, time=t)
writeOutputMeshFunctions()





### output time functions ###
t_i=0
t_iOutput = t_i
outfile_timeFunctions = File(output_dir_path + "/../data/temp/timeFunctions.pvd")

numberOfOutputTimeSteps = np.ceil((T_end-T_0)/writeOutputEvery).astype(int)
numberOfValuesInTimeFunctions = numberOfOutputTimeSteps+1

meshTime = IntervalMesh(numberOfOutputTimeSteps, T_0, T_end)
VecSpaceTime = VectorFunctionSpace(meshTime, "CG", 1)

timeValuesTime = np.zeros(numberOfValuesInTimeFunctions)
timeValuesTime[t_iOutput] = T_0
TimeFunctionTime = Function(VecSpaceTime,timeValuesTime[:],"time")

L2timeValuesTheta = np.zeros(numberOfValuesInTimeFunctions)
L2timeValuesTheta[t_iOutput] = norm(theta,"l2")
L2normTimeFunctionTheta = Function(VecSpaceTime,L2timeValuesTheta[:],"||theta||")

L2timeValuesGradTheta = np.zeros(numberOfValuesInTimeFunctions)
L2timeValuesGradTheta[t_iOutput] = norm(project(grad(theta),V_vec),"l2")
L2normTimeFunctionGradTheta = Function(VecSpaceTime,L2timeValuesGradTheta[:],"||grad theta||")

Hminus1timeValuesTheta = np.zeros(numberOfValuesInTimeFunctions)
Hminus1timeValuesTheta[t_iOutput] = calcHminus1NormOfScalar(theta)
Hminus1normTimeFunctionTheta = Function(VecSpaceTime,Hminus1timeValuesTheta[:],"||grad^-1 theta||")

Hminus1OverL2timeValuesTheta = np.zeros(numberOfValuesInTimeFunctions)
Hminus1OverL2timeValuesTheta[t_iOutput] = Hminus1timeValuesTheta[t_iOutput]/L2timeValuesGradTheta[t_iOutput]
Hminus1OverL2TimeFunctionTheta = Function(VecSpaceTime,Hminus1OverL2timeValuesTheta[:],"||grad^-1 theta||/||theta||")

L2TimeValuesU_adv = np.zeros(numberOfValuesInTimeFunctions)
L2TimeValuesU_adv[t_iOutput] = norm(u_adv,"l2")
L2normTimeFunctionU_adv = Function(VecSpaceTime,L2TimeValuesU_adv[:],"||u_adv||")


calcConstants_batscaleGammaUmiles()
print("l_dom", l_dom)
print("l_root", l_root)
print("l_bat", l_bat)
        
lDomTimeValues = np.zeros(numberOfValuesInTimeFunctions)
lDomTimeValues[t_iOutput] = l_dom
lDomTimeFunction = Function(VecSpaceTime,lDomTimeValues[:],"l_dom")

lBatTimeValues = np.zeros(numberOfValuesInTimeFunctions)
lBatTimeValues[t_iOutput] = l_bat
lBatTimeFunction = Function(VecSpaceTime,lBatTimeValues[:],"l_bat")

lRootTimeValues = np.zeros(numberOfValuesInTimeFunctions)
lRootTimeValues[t_iOutput] = l_root
lRootTimeFunction = Function(VecSpaceTime,lRootTimeValues[:],"l_rootLapLap2")

outfile_timeFunctions.write(TimeFunctionTime,L2normTimeFunctionTheta, L2normTimeFunctionGradTheta, Hminus1normTimeFunctionTheta, Hminus1OverL2TimeFunctionTheta, L2normTimeFunctionU_adv, lBatTimeFunction, lDomTimeFunction, lRootTimeFunction, time=t)







### copy script to save it ####
copy(os.path.realpath(__file__), output_dir_path + "/../data/temp/0used_script.py")

### simulating ###

timeStartSolving = datetime.datetime.now()
lastRealTime = timeStartSolving
lastWrittenOutput = 0
while (t < T_end):
    solver.solve()
    t_i += 1
    t += timestep
    
    
    if numberTestFunctions == 1:
        theta_old.assign(theta)
    else:
        w_old.assign(w)
        theta, theta_laplace = w.split()
        
    if forceZeroAverage:
        theta = getZeroAverageOfScalarFunction(theta)
        
    if usedFlow in ['milesEnergy']:
        if(t>=timeInitialUadv):
            milesOptimalFlowEnergyCase = calcMilesOptimalFlowEnergyCase(theta)
            u_adv.assign(milesOptimalFlowEnergyCase)
    elif usedFlow in ['milesEnstrophy']:
        if(t>=timeInitialUadv):
            milesOptimalFlowEnstrophyCase = calcMilesOptimalFlowEnstrophyCase(theta)
            u_adv.assign(milesOptimalFlowEnstrophyCase)
    elif usedFlow in ['camillaTestFlow']:
        if(t>=timeInitialUadv):
            camillaTestFlow = calcCamillaTestFlow(theta)
            u_adv.assign(camillaTestFlow)
    
    
    
    
    ##### outputs ###
    if t >= lastWrittenOutput + writeOutputEvery:
        lastWrittenOutput = t
        t_iOutput += 1
        ### write output time functions ###
        timeValuesTime[t_iOutput] = t
        TimeFunctionTime = Function(VecSpaceTime,timeValuesTime[:],"time")
    
        L2normTheta = norm(theta,"l2")
        L2timeValuesTheta[t_iOutput] = L2normTheta
        L2normTimeFunctionTheta = Function(VecSpaceTime,L2timeValuesTheta[:],"||theta||")
    
        L2timeValuesGradTheta[t_iOutput] = norm(project(grad(theta),V_vec),"l2")
        L2normTimeFunctionGradTheta = Function(VecSpaceTime,L2timeValuesGradTheta[:],"||grad theta||")
    
        Hminus1normTheta = calcHminus1NormOfScalar(theta)
        Hminus1timeValuesTheta[t_iOutput] = Hminus1normTheta
        Hminus1normTimeFunctionTheta = Function(VecSpaceTime,Hminus1timeValuesTheta[:],"||grad^-1 theta||")
        
        Hminus1OverL2timeValuesTheta[t_iOutput] = Hminus1normTheta/L2normTheta
        Hminus1OverL2TimeFunctionTheta = Function(VecSpaceTime,Hminus1OverL2timeValuesTheta[:],"||grad^-1 theta||/||theta||")
        
        L2TimeValuesU_adv[t_iOutput] = norm(u_adv,"l2")
        L2normTimeFunctionU_adv = Function(VecSpaceTime,L2TimeValuesU_adv[:],"||u_adv||")
        
        
        calcConstants_batscaleGammaUmiles()
        
        lDomTimeValues[t_iOutput] = l_dom
        lDomTimeFunction = Function(VecSpaceTime,lDomTimeValues[:],"l_dom")
        
        lBatTimeValues[t_iOutput] = l_bat
        lBatTimeFunction = Function(VecSpaceTime,lBatTimeValues[:],"l_bat")
        
        lRootTimeValues[t_iOutput] = l_root
        lRootTimeFunction = Function(VecSpaceTime,lRootTimeValues[:],"l_rootLapLap2")




        outfile_timeFunctions.write(TimeFunctionTime,L2normTimeFunctionTheta, L2normTimeFunctionGradTheta, Hminus1normTimeFunctionTheta, Hminus1OverL2TimeFunctionTheta, L2normTimeFunctionU_adv, lBatTimeFunction, lDomTimeFunction, lRootTimeFunction, time=t)
        #outfile_timeFunctions.write(TimeFunctionTime,L2normTimeFunctionTheta, L2normTimeFunctionGradTheta, Hminus1normTimeFunctionTheta, L2normTimeFunctionU_adv, time=t)
        #outfile_timeFunctions.write(project(TimeFunctionTime, VecSpaceTime, name="time"),project(L2normTimeFunctionTheta, VecSpaceTime, name="theta L^2"), project(L2normTimeFunctionGradTheta, VecSpaceTime, name="grad theta L^2"), time=t)


    
        ### write output mesh functions ###   
        writeOutputMeshFunctions()





    print(np.round(t_i/numberOfTimesteps*100,2),"% ( step = ", t_i, " von ", numberOfTimesteps,", time t = ", np.round(t,4),") after ", datetime.datetime.now()-lastRealTime, ", estimated time left ", ((T_end-T_0)/t-1)*(datetime.datetime.now()-timeStartSolving)  )
    lastRealTime = datetime.datetime.now()

time_end = datetime.datetime.now()
print("ending at ",time_end)
print("total time ", time_end-time_start)

