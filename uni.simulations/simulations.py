from firedrake import *
import numpy as np
import os

### time ###
import time
import datetime

### code source code file ###
from shutil import copy
from ufl.tensors import as_scalar
from ufl.operators import div, grad, nabla_grad
from ufl import indexed
from cmath import sqrt
from firedrake.utility_meshes import PeriodicIntervalMesh
from ufl.differentiation import NablaGrad
from numpy import imag



# using now() to get current time  
time_start = datetime.datetime.now()
print(time_start,"starting ...")  





### PARAMETERS ###

# interval lengths
L = 128                 #128
L_x = L

# spatial steps
nProL = 8               #8
n_x = L_x*nProL

# times
numberOfTimestepsPerUnit = 200
T_end = 1000
timeInitialUadv = 0.001      ### for miles u_adv need a sine flow until t = 0.01 (otherwise get a stationary solution)

# pde name
# list of available shortNames: burger/burgers/nonLin, advLap, viscBurgers/burgersHeat/burgerHeat/burgersLap/burgerLap, burgersLap2, KdVKuraSiva
pdeShortName = "KdVKuraSiva"

# finite elements
finitEleFamily = "CG"           #https://www.firedrakeproject.org/variational-problems.html#supported-finite-elements
finitEleDegree = 1


# force 0 average after every step? True, False12
forceZeroAverage = False

# kappa in theta_t + < u_adv, grad theta> + kappa*laplace theta + laplace^2 theta = 0
kappa = 1

print(kappa)

### initial condition ###
ic_scale = 1
ic_freq = 1
ic_scale_x = ic_scale
ic_freq_x = ic_freq
randomIC = False
# possible initial data files 20210929_162000_1024Random1Durch1000Values, 20210929_162000_4096Random1Durch1000Values
loadInitialDataFilename = ""


### rescale outputs -> \| . \|_2 = 1 ###
rescaleOutputs = True
inverseLaplacianEnforceAverageFreeBefore = True
inverseLaplacianEnforceAverageFreeAfter = True



### write output only every ... time intervals to save storage space
writeOutputEvery = 0.1             # 0 -> every time,
 
### PARAMETERS END ###















### settings ###
maxLogOutputsPerSecond = 1


### files
mashFunctionsFilePath = "/../data/temp/u.pvd"
timeFunctionsFilePath = "/../data/temp/timeFunctions.pvd"



















output_dir_path = os.path.dirname(os.path.realpath(__file__))
outfile_u = File(output_dir_path + mashFunctionsFilePath)
outfile_timeFunctions = File(output_dir_path + timeFunctionsFilePath)


T_0 = 0
t = T_0
numberOfTimesteps = round(numberOfTimestepsPerUnit*(T_end-T_0))
timestepInitial = (T_end-T_0)/numberOfTimesteps
timestep = timestepInitial

# mesh = UnitSquareMesh(n, n)
#mesh = PeriodicUnitSquareMesh(n,n)
mesh = PeriodicIntervalMesh(n_x,L_x)

V = VectorFunctionSpace(mesh, finitEleFamily, finitEleDegree)
V_vec = VectorFunctionSpace(mesh, finitEleFamily, finitEleDegree)
V_out = VectorFunctionSpace(mesh, finitEleFamily, finitEleDegree)
x = SpatialCoordinate(mesh)


print(datetime.datetime.now(),"spaces defined")


### initial data ###
x_u0 = SpatialCoordinate(mesh)

def posPartofFunction(function):
    return 1/2*(np.abs(function)+function)
    
ic_c = 1
ic_u = project(as_vector([ic_scale*sin(ic_freq_x*2*pi*x_u0[0]/L_x)]), V)
#ic_u = project(as_vector([ic_scale*posPartofFunction(x_u0[0]*sin(ic_freq_x*2*pi*x_u0[0]/L_x))]), V)
#ic_u = project(as_vector([ np.power(np.exp(1),(-np.power(1/2*(x_u0[0]-50),2))) ]), V)
#ic_u = project(as_vector([ic_scale*cos(ic_freq_x*2*pi*x_u0[0]/L_x)]), V)



u_0Random = 1/1000*2*(np.random.rand(n_x)-0.5)       # zwischen 0 und 1

if randomIC:
    ic_u = Function(V,u_0Random)

### advection term ###
u_adv = Function(V)
u_adv.assign(0)

if len(loadInitialDataFilename)>0:
    u0_loaded = np.load(output_dir_path + "/../data/initialData/" + loadInitialDataFilename +".npy")
    print(datetime.datetime.now(),"loading initial data " ,"\t", "/data/initialData/" + loadInitialDataFilename +".npy")
    ic_u = Function(V,u0_loaded)

    
    
print(datetime.datetime.now(),"inital values assigned ")  





### functions ###
###################################
def getZeroAverageOfScalarFunction(function):
    if isinstance(function, Function):
        thisFunction = function
    else:
        thisFunction = project(function,V)      ### when indexed for example theta, theta_laplace = w.split() then theta is not a function but rather Indexed: w_20[0] or something like this
    sum = np.sum(thisFunction.dat.data)
    thisFunction.dat.data[:] = thisFunction.dat.data[:]-sum/(n_x)
    return thisFunction
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
        if norm(getZeroAverageOfScalarFunction(function)-function,"l2")>0.01*((L_x)**0.5):
            print("!!!warning!!! initial data of get inverse laplacian is non average free -> enforcing average free")
        function = getZeroAverageOfScalarFunction(function)
    
    outFunction = Function(V)
    testFunctionInvLap = TestFunction(V)
    F_invLap = (inner(grad(outFunction),grad(testFunctionInvLap))+dot(function,testFunctionInvLap))*dx
    solve(F_invLap == 0, outFunction)
    if inverseLaplacianEnforceAverageFreeAfter:
        if norm(getZeroAverageOfScalarFunction(outFunction)-outFunction,"l2")>0.01*((L_x)**0.5):
            print("!!!warning!!! result of get inverse laplacian is non average free -> enforcing average free")
        outFunction = getZeroAverageOfScalarFunction(outFunction)
    return outFunction
###################################
def calcLaplacian(function):
    gradFunction = project(grad(function), V_vec)
    return project(div(gradFunction),V)
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
def calcDivMinus1ofScalar(function):
    # div^{-1} = div^{-1} divgrad laplace^{-1} = grad laplace^{-1}
    return grad(calcInverseLaplacian(function))
def calcHminus1NormOfScalar(function):
    return norm(calcDivMinus1ofScalar(function),"l2")
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
    outU = Function(V_out, u, "u")
    outfile_u.write(u, time=t)





    
    

if pdeShortName in ['nonLin','burger','burgers']:
    pdeShortName = 'nonLin'
    
if pdeShortName in ['viscBurgers','burgersHeat','burgerHeat','burgersLap','burgerLap']:
    pdeShortName = 'burgersLap'
    
    
if pdeShortName in ['nonLin']:
    numberTestFunctions = 1
elif pdeShortName in ['onlyAdv']:
    numberTestFunctions = 1
elif pdeShortName in ['advLap']:
    numberTestFunctions = 1
elif pdeShortName in ['burgersLap']:
    numberTestFunctions = 1
elif pdeShortName in ['advLap2Lap']:
    numberTestFunctions = 2
elif pdeShortName in ['advLap2']:
    numberTestFunctions = 2
elif pdeShortName in ['burgersLap2']:
    numberTestFunctions = 2
elif pdeShortName in ['kuraSiva']:
    numberTestFunctions = 2
elif pdeShortName in ['KdVKuraSiva']:
    numberTestFunctions = 2
    
else:
    print("ERROR wrong pde short name (not in list)")

### define the functions
if numberTestFunctions == 1:
    u = Function(V)
    u_old = Function(V)
    
    u.assign(ic_u)
    u_old.assign(ic_u)

    u_laplace = Function(V)
    u_laplace_old = Function(V)
    testFunctionA = TestFunction(V)
    writeOutputMeshFunctions()

else:
    W = MixedFunctionSpace((V, V))
    W_out = MixedFunctionSpace((V_out, V_out))
    
    w = Function(W)
    u, u_laplace = w.split()
    
    w_old = Function(W)
    u_old, u_laplace_old = w_old.split()
    
    ### not sure about initial condition for laplace u = 0 since it would have to be laplace ic_u
    ### but should (and looks like does) work via the pde 
    
    u.assign(ic_u)
    u_old.assign(ic_u)
    writeOutputMeshFunctions()   
    
    
    u, u_laplace = split(w)
    u_old, u_laplace_old = split(w_old)
    
    
    
    testFunctionA, testFunctionB = TestFunctions(W)
#################################





##### pde #####

if numberTestFunctions == 1:
    F_nonLin = (inner((u - u_old)/timestep, testFunctionA)
                + inner(dot(u,grad(u)), testFunctionA)
                )*dx
    
    F_onlyAdv = (inner((u - u_old)/timestep, testFunctionA)
                 + inner(dot(u_adv,grad(u)), testFunctionA)
                 )*dx

    F_advLap = (inner((u - u_old)/timestep, testFunctionA)
                + inner(dot(u_adv,grad(u)), testFunctionA) 
                + kappa * inner(grad(u), grad(testFunctionA))
                )*dx
                
    F_burgersLap = (inner((u - u_old)/timestep, testFunctionA)
                + inner(dot(u,grad(u)), testFunctionA)
                + kappa * inner(grad(u), grad(testFunctionA))
                )*dx

if numberTestFunctions == 2:
    F_advLap2Lap = (inner((u - u_old)/timestep, testFunctionA)
        + inner(dot(u_adv,grad(u)), testFunctionA) 
        + kappa*inner(u_laplace, testFunctionA)
        - kappa*inner(grad(u_laplace), grad(testFunctionA))
        + inner(u_laplace, testFunctionB)
        + inner(grad(u), grad(testFunctionB))
        )*dx
    
    F_advLap2 = (inner((u - u_old)/timestep, testFunctionA)
        + inner(dot(u_adv,grad(u)), testFunctionA)
        - kappa*inner(grad(u_laplace), grad(testFunctionA))
        + inner(u_laplace, testFunctionB)
        + inner(grad(u), grad(testFunctionB))
        )*dx
        
    
    F_burgersLap2 = (inner((u - u_old)/timestep, testFunctionA)
        + inner(dot(u,nabla_grad(u)), testFunctionA)
        - kappa*inner(nabla_grad(u_laplace), grad(testFunctionA))
        + inner(u_laplace, testFunctionB)
        + inner(nabla_grad(u), grad(testFunctionB))
        )*dx
        
    F_kuraSiva = (inner((u - u_old)/timestep, testFunctionA)
        + inner(dot(u,nabla_grad(u)), testFunctionA)
        + kappa*inner(u_laplace, testFunctionA)
        - kappa*inner(grad(u_laplace), grad(testFunctionA))
        + inner(u_laplace, testFunctionB)
        + inner(grad(u), grad(testFunctionB))
        )*dx
        
    epsilonKdVKuraSiva = np.sqrt(1-np.power(kappa,2))
    F_KdVKuraSiva = (inner((u - u_old)/timestep, testFunctionA)
        + inner(dot(u,nabla_grad(u)), testFunctionA)
        + kappa*inner(u_laplace, testFunctionA)
        - kappa*inner(grad(u_laplace), grad(testFunctionA))
        + abs(epsilonKdVKuraSiva)*inner(u_laplace.dx(0), testFunctionA)
        + inner(u_laplace, testFunctionB)
        + inner(grad(u), grad(testFunctionB))
        )*dx

if pdeShortName in ['nonLin']:
    F = F_nonLin
elif pdeShortName in ['onlyAdv']:
    F = F_onlyAdv
elif pdeShortName in ['advLap']:
    F = F_advLap
elif pdeShortName in ['burgersLap']:
    F = F_burgersLap
elif pdeShortName in ['advLap2Lap']:
    F = F_advLap2Lap
elif pdeShortName in ['advLap2']:
    F = F_advLap2
elif pdeShortName in ['burgersLap2']:
    F = F_burgersLap2
elif pdeShortName in ['kuraSiva']:
    F = F_kuraSiva
elif pdeShortName in ['KdVKuraSiva']:
    F = F_KdVKuraSiva
    if epsilonKdVKuraSiva != abs(epsilonKdVKuraSiva):
        print(datetime.datetime.now(), "ERROR\t be careful\t epsilon is not real -> taking the absolute value (probably kappa>1)")    # gibt automatisch error weil sqrt(1-kappa²)        
    print(datetime.datetime.now(),"epsilon = ",epsilonKdVKuraSiva)
else:
    print("ERROR wrong pde short name (not in list)")


# problem = NonlinearVariationalProblem(F, w, bcs=[bc_u, bc_v])
if numberTestFunctions == 1:
    problem = NonlinearVariationalProblem(F, u)
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








### output time functions ###
t_i=0
t_iOutput = t_i
numberOfOutputTimeSteps = np.ceil((T_end-T_0)/writeOutputEvery).astype(int)
numberOfValuesInTimeFunctions = numberOfOutputTimeSteps+1

meshTime = IntervalMesh(numberOfOutputTimeSteps, T_0, T_end)
VecSpaceTime = VectorFunctionSpace(meshTime, "CG", 1)

timeValuesTime = np.zeros(numberOfValuesInTimeFunctions)
timeValuesTime[t_iOutput] = T_0
TimeFunctionTime = Function(VecSpaceTime,timeValuesTime[:],"time")

L2timeValuesU = np.zeros(numberOfValuesInTimeFunctions)
L2normU = norm(u,"l2")
L2timeValuesU[t_iOutput] = L2normU
L2normTimeFunctionU = Function(VecSpaceTime,L2timeValuesU[:],"||u||")

L2timeValuesGradU = np.zeros(numberOfValuesInTimeFunctions)
L2timeValuesGradU[t_iOutput] = norm(grad(u),"l2")
L2normTimeFunctionGradU = Function(VecSpaceTime,L2timeValuesGradU[:],"||grad u||")

Hminus1timeValuesU = np.zeros(numberOfValuesInTimeFunctions)
Hminus1normU = 0#calcHminus1NormOfScalar(u)
Hminus1timeValuesU[t_iOutput] = Hminus1normU
Hminus1normTimeFunctionU = Function(VecSpaceTime,Hminus1timeValuesU[:],"||grad^-1 u||")
    
Hminus1OverL2timeValuesU = np.zeros(numberOfValuesInTimeFunctions)
Hminus1OverL2timeValuesU[t_iOutput] = Hminus1normU/L2normU
Hminus1OverL2TimeFunctionU = Function(VecSpaceTime,Hminus1OverL2timeValuesU[:],"||grad^-1 u||/||u||")

L2TimeValuesU_adv = np.zeros(numberOfValuesInTimeFunctions)
L2TimeValuesU_adv[t_iOutput] = norm(u_adv,"l2")
L2normTimeFunctionU_adv = Function(VecSpaceTime,L2TimeValuesU_adv[:],"||u_adv||")


outfile_timeFunctions.write(TimeFunctionTime,L2normTimeFunctionU, L2normTimeFunctionGradU, Hminus1normTimeFunctionU, Hminus1OverL2TimeFunctionU, L2normTimeFunctionU_adv, time=t)







### copy script to save it ####
copy(os.path.realpath(__file__), output_dir_path + "/../data/temp/0used_script.py")

### simulating ###

timeStartSolving = datetime.datetime.now()
lastRealTime = timeStartSolving
lastWrittenOutput = 0
lastLogOutput = 0
while (t < T_end):
    solver.solve()
    t_i += 1
    t += timestep
    
    #cFunktion.interpolate(sin(2*pi*t)*sin(2*pi*x/L_x)+cos(2*pi*t)*sin(2*pi*y/L_y))
    
    if numberTestFunctions == 1:
        u_old.assign(u)
    else:
        w_old.assign(w)
        u, u_laplace = w.split()
        
    if forceZeroAverage:
        u = getZeroAverageOfScalarFunction(u)
        
    
    
    
    
    ##### outputs ###
    if t >= lastWrittenOutput + writeOutputEvery:
        lastWrittenOutput = t
        t_iOutput += 1
        ### write output time functions ###
        timeValuesTime[t_iOutput] = t
        TimeFunctionTime = Function(VecSpaceTime,timeValuesTime[:],"time")
    
        L2normU = norm(u,"l2")
        L2timeValuesU[t_iOutput] = L2normU
        L2normTimeFunctionU = Function(VecSpaceTime,L2timeValuesU[:],"||u||")
    
        L2timeValuesGradU[t_iOutput] = norm(grad(u),"l2")
        L2normTimeFunctionGradU = Function(VecSpaceTime,L2timeValuesGradU[:],"||grad u||")
    
        Hminus1normU = 0#calcHminus1NormOfScalar(u)
        Hminus1timeValuesU[t_iOutput] = Hminus1normU
        Hminus1normTimeFunctionU = Function(VecSpaceTime,Hminus1timeValuesU[:],"||grad^-1 u||")
        
        Hminus1OverL2timeValuesU[t_iOutput] = Hminus1normU/L2normU
        Hminus1OverL2TimeFunctionU = Function(VecSpaceTime,Hminus1OverL2timeValuesU[:],"||grad^-1 u||/||u||")
        
        L2TimeValuesU_adv[t_iOutput] = norm(u_adv,"l2")
        L2normTimeFunctionU_adv = Function(VecSpaceTime,L2TimeValuesU_adv[:],"||u_adv||")
        

        outfile_timeFunctions.write(TimeFunctionTime,L2normTimeFunctionU, L2normTimeFunctionGradU, Hminus1normTimeFunctionU, Hminus1OverL2TimeFunctionU, L2normTimeFunctionU_adv, time=t)

    
        ### write output mesh functions ###   
        writeOutputMeshFunctions()
        #uOutput = Function(V, u, "u")
        #outfile_u.write(uOutput, time=t)





    if (time.time()-lastLogOutput)>maxLogOutputsPerSecond:
        print(np.round(t_i/numberOfTimesteps*100,2),"% ( step = ", t_i, " of ", numberOfTimesteps,", time t = ", np.round(t,4),") after ", datetime.datetime.now()-lastRealTime, ", estimated time left ", ((T_end-T_0)/t-1)*(datetime.datetime.now()-timeStartSolving)  )
        lastLogOutput = time.time()
    lastRealTime = datetime.datetime.now()

time_end = datetime.datetime.now()
print("ending at ",time_end)
print("total time ", time_end-time_start)

