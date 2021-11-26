import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"             # had a warning that it might run best on 1 thread
from firedrake import *
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
infoString = "simulation info"
infoString += "\n\t"+"time_start"+" = \t\t"+str(time_start)





### PARAMETERS ###

# interval lengths
L = 128                 #128
 
# spatial steps
nProL = 8               #8
n = L*nProL

# times
numberOfTimestepsPerUnit = 200
T_end = 3000
timeInitialUadv = 0.001      ### for miles u_adv need a sine flow until t = 0.01 (otherwise get a stationary solution)

# pde name
# list of available shortNames: 
    #KdVKuraSiva
pdeShortName = "KdVKuraSiva"

# finite elements
finitEleFamily = "CG"           #https://www.firedrakeproject.org/variational-problems.html#supported-finite-elements
finitEleDegree = 1


# force 0 average of the initial data? True, False
forceICZeroAverage = False

# kappa
#in theta_t + < u_adv, grad theta> + kappa*laplace theta + laplace^2 theta = 0
kappa = 1/16
epsilonKdVKuraSiva = np.sqrt(1-np.power(kappa,2))       # np.sqrt(1-np.power(kappa,2))


### initial condition ###
ic_scale = 1
ic_freq = 1
ic_freqShift = 0
randomIC = False
# possible initial data files:
    # 20210929_162000_1024Random1Durch1000Values, 20210929_162000_4096Random1Durch1000Values
    # 20211026_140000_1024Random1Durch1Values, 20211026_140000_4096Random1Durch1Values
    # 20211028_113529_1024Random_scale1 
loadInitialDataFilename = "20210929_162000_1024Random1Durch1000Values"


### rescale outputs -> \| . \|_2 = 1 ###
rescaleOutputs = True



### write output only every ... time intervals to save storage space
writeOutputEvery = 1             # 0 -> every time,

overwritePreviousData = False

writeTimeFunctionsOnlyAtTheEnd = True           # write output of time functions only at the end to save writing (-> simulation) time
### PARAMETERS END ###























infoString += "\n\t"+"L"+" = \t\t\t\t"+str(L)
infoString += "\n\t"+"nProL"+" = \t\t"+str(nProL)
infoString += "\n\t"+"numberOfTimestepsPerUnit"+" = \t\t"+str(numberOfTimestepsPerUnit)
infoString += "\n\t"+"T_end"+" = \t\t"+str(T_end)
infoString += "\n\t"+"timeInitialUadv"+" = \t\t"+str(timeInitialUadv)
infoString += "\n\t"+"pdeShortName"+" = \t\t"+str(pdeShortName)
infoString += "\n\t"+"finitEleFamily"+" = \t\t"+str(finitEleFamily)
infoString += "\n\t"+"finitEleDegree"+" = \t\t"+str(finitEleDegree)
infoString += "\n\t"+"forceICZeroAverage"+" = \t\t"+str(forceICZeroAverage)
infoString += "\n\t"+"kappa"+" = \t\t"+str(kappa)
infoString += "\n\t"+"epsilonKdVKuraSiva"+" = \t\t"+str(epsilonKdVKuraSiva)
infoString += "\n\t"+"ic_scale"+" = \t\t"+str(ic_scale)
infoString += "\n\t"+"ic_freq"+" = \t\t"+str(ic_freq)
infoString += "\n\t"+"randomIC"+" = \t\t"+str(randomIC)
infoString += "\n\t"+"loadInitialDataFilename"+" = \t\t"+str(loadInitialDataFilename)
infoString += "\n\t"+"rescaleOutputs"+" = \t\t"+str(rescaleOutputs)
infoString += "\n\t"+"writeOutputEvery"+" = \t\t"+str(writeOutputEvery)










### settings ###
maxLogOutputsPerSecond = 1


### files
scriptPath = os.path.dirname(os.path.realpath(__file__))
output_dir_path = scriptPath+"/../data/temp/"
initialDataPath = scriptPath+"/initialData/"
meshFunctionsFilePath = output_dir_path+"simulationData/u.pvd"
timeFunctionsFilePath = output_dir_path+"simulationData/timeFunctions.pvd"
infoFilePath = output_dir_path+"info.txt"
# check if cleaned up output folder
if not(overwritePreviousData) and os.path.isfile(infoFilePath):
    print(output_dir_path)
    raise Exception(infoFilePath + ' found - probably not cleaned up the output directory:\n\txdg-open '+output_dir_path)
outfile_u = File(meshFunctionsFilePath)
outfile_timeFunctions = File(timeFunctionsFilePath)



infoString += "\n\t"+"meshFunctionsFilePath"+" = \t\t"+str(meshFunctionsFilePath)
infoString += "\n\t"+"timeFunctionsFilePath"+" = \t\t"+str(timeFunctionsFilePath)


















T_0 = 0
t = T_0
numberOfTimesteps = round(numberOfTimestepsPerUnit*(T_end-T_0))
timestepInitial = (T_end-T_0)/numberOfTimesteps
timestep = timestepInitial

mesh = PeriodicIntervalMesh(n,L)

V = FunctionSpace(mesh, finitEleFamily, finitEleDegree)
V_vec = FunctionSpace(mesh, finitEleFamily, finitEleDegree)
V_out = FunctionSpace(mesh, "DG", 0)
x = SpatialCoordinate(mesh)


print(datetime.datetime.now(),"spaces defined")


### initial data ###
x_u0 = SpatialCoordinate(mesh)

ic_c = 1
ic_u = Function(V)
ic_u.interpolate(ic_scale*sin(ic_freq*2*pi*x_u0[0]/L+ic_freqShift))
#ic_u = project(as_vector([ic_scale*posPartofFunction(x_u0[0]*sin(ic_freq_x*2*pi*x_u0[0]/L_x))]), V)
#ic_u = project(as_vector([ np.power(np.exp(1),(-np.power(1/2*(x_u0[0]-50),2))) ]), V)
#ic_u = project(as_vector([ic_scale*cos(ic_freq_x*2*pi*x_u0[0]/L_x)]), V)



u_0Random = 1/1000*2*(np.random.rand(n)-0.5)       # zwischen 0 und 1

if randomIC:
    ic_u = Function(V,u_0Random)


if len(loadInitialDataFilename)>0:
    initialDataLoadingPath = initialDataPath + loadInitialDataFilename +".npy"
    infoString += "\n\t"+"initialDataLoadingPath"+" = \t\t"+str(initialDataLoadingPath)
    u0_loaded = np.load(initialDataLoadingPath)
    print(datetime.datetime.now(),"loading initial data " ,"\t", initialDataLoadingPath)
    ic_u = Function(V,u0_loaded)

    
    
print(datetime.datetime.now(),"inital values assigned ")  









    
    

if pdeShortName in ['nonLin','burger','burgers']:
    pdeShortName = 'nonLin'
    
if pdeShortName in ['viscBurgers','burgersHeat','burgerHeat','burgersLap','burgerLap']:
    pdeShortName = 'burgersLap'
    
    
if pdeShortName in ['xyz']:
    numberTestFunctions = 1
elif pdeShortName in ['KdVKuraSiva']:
    numberTestFunctions = 2
    
else:
    print("ERROR wrong pde short name (not in list)")
    
    
def writeMeshFunctions():
    outfile_u.write(project(u, V_out,name="u"), time=t)


if forceICZeroAverage:
    ic_u -= assemble(ic_u*dx)/L

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
    writeMeshFunctions()   
    
    
    u, u_laplace = split(w)
    u_old, u_laplace_old = split(w_old)
    
    
    
    testFunctionA, testFunctionB = TestFunctions(W)
#################################





##### pde #####

#if numberTestFunctions == 1:
#    F_nonLin = (inner((u - u_old)/timestep, testFunctionA)
#                + inner(dot(u,grad(u)), testFunctionA)
#                )*dx
#    
#    F_onlyAdv = (inner((u - u_old)/timestep, testFunctionA)
#                 + inner(dot(u_adv,grad(u)), testFunctionA)
#                 )*dx
#
#    F_advLap = (inner((u - u_old)/timestep, testFunctionA)
#                + inner(dot(u_adv,grad(u)), testFunctionA) 
#                + kappa * inner(grad(u), grad(testFunctionA))
#                )*dx
#                
#    F_burgersLap = (inner((u - u_old)/timestep, testFunctionA)
#                + inner(dot(u,grad(u)), testFunctionA)
#                + kappa * inner(grad(u), grad(testFunctionA))
#                )*dx
#                
#    F_test1function = (inner((u - u_old)/timestep, testFunctionA)
#                + abs(epsilonKdVKuraSiva)*inner(u.dx(0), testFunctionA)
#                )*dx

if numberTestFunctions == 2:
#    F_advLap2Lap = (inner((u - u_old)/timestep, testFunctionA)
#        + inner(dot(u_adv,grad(u)), testFunctionA) 
#        + kappa*inner(u_laplace, testFunctionA)
#        - kappa*inner(grad(u_laplace), grad(testFunctionA))
#        + inner(u_laplace, testFunctionB)
#        + inner(grad(u), grad(testFunctionB))
#        )*dx
    
#    F_advLap2 = (inner((u - u_old)/timestep, testFunctionA)
#        + inner(dot(u_adv,grad(u)), testFunctionA)
#        - kappa*inner(grad(u_laplace), grad(testFunctionA))
#        + inner(u_laplace, testFunctionB)
#        + inner(grad(u), grad(testFunctionB))
#        )*dx
#        
#    
#    F_burgersLap2 = (inner((u - u_old)/timestep, testFunctionA)
#        + inner(dot(u,nabla_grad(u)), testFunctionA)
#        - kappa*inner(nabla_grad(u_laplace), grad(testFunctionA))
#        + inner(u_laplace, testFunctionB)
#        + inner(nabla_grad(u), grad(testFunctionB))
#        )*dx
#        
#    F_kuraSiva = (inner((u - u_old)/timestep, testFunctionA)
#        + inner(dot(u,nabla_grad(u)), testFunctionA)
#        + kappa*inner(u_laplace, testFunctionA)
#        - kappa*inner(grad(u_laplace), grad(testFunctionA))
#        + inner(u_laplace, testFunctionB)
#        + inner(grad(u), grad(testFunctionB))
#        )*dx
#   
     
    F_KdVKuraSiva = (
        inner((u - u_old)/timestep, testFunctionA)
        + inner(dot(u,u.dx(0)), testFunctionA)
        + kappa*inner(u_laplace, testFunctionA)
        - kappa*inner(grad(u_laplace), grad(testFunctionA))
        + abs(epsilonKdVKuraSiva)*inner(u_laplace.dx(0), testFunctionA)
        + inner(u_laplace, testFunctionB)
        + inner(grad(u), grad(testFunctionB))
        )*dx
        
#    F_test2functions = (inner((u - u_old)/timestep, testFunctionA)
#        + abs(epsilonKdVKuraSiva)*inner(u_laplace.dx(0), testFunctionA)
#        + inner(u_laplace, testFunctionB)
#        + inner(grad(u), grad(testFunctionB))
#        )*dx

if pdeShortName in ['nonLin']:
    F = F_nonLin
elif pdeShortName in ['onlyAdv']:
    F = F_onlyAdv
elif pdeShortName in ['advLap']:
    F = F_advLap
elif pdeShortName in ['burgersLap']:
    F = F_burgersLap
elif pdeShortName in ['test1function']:
    F = F_test1function
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
elif pdeShortName in ['test2functions']:
    F = F_test2functions
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
t_iOutput = 0
numberOfOutputTimeSteps = np.ceil((T_end-T_0)/writeOutputEvery).astype(int)
numberOfValuesInTimeFunctions = numberOfOutputTimeSteps+1

meshTime = IntervalMesh(numberOfOutputTimeSteps, T_0, T_end)
functionSpaceTime = FunctionSpace(meshTime, "CG", 1)

timeValuesTime = np.zeros(numberOfValuesInTimeFunctions)
L2timeValuesU = np.zeros(numberOfValuesInTimeFunctions)
intTimeValuesU = np.zeros(numberOfValuesInTimeFunctions)
L2timeValuesGradU = np.zeros(numberOfValuesInTimeFunctions)
L2timeValuesLaplaceU = np.zeros(numberOfValuesInTimeFunctions)
intUUxhoch2ValuesU = np.zeros(numberOfValuesInTimeFunctions)
kdvSpeed = np.zeros(numberOfValuesInTimeFunctions)
kdvSpeedInParaview = np.zeros(numberOfValuesInTimeFunctions)


def saveTimeFunctionsArray(index=t_iOutput):
    timeValuesTime[index] = t
    L2timeValuesU[index] = norm(u,"l2")
    intTimeValuesU[index] = assemble(u*dx)
    L2timeValuesGradU[index] = norm(grad(u),"l2")
    L2timeValuesLaplaceU[index] = norm(u_laplace,"l2")
    intUUxhoch2ValuesU[index] = assemble(u*u.dx(0)*u.dx(0)*dx)
    kdvSpeed[index] = (intUUxhoch2ValuesU[index]-epsilonKdVKuraSiva*L2timeValuesLaplaceU[index]*L2timeValuesLaplaceU[index])/(L2timeValuesGradU[index]*L2timeValuesGradU[index])
    kdvSpeedInParaview[index] = kdvSpeed[index]*(n*2)/L         ### WARNING n*2 seems to be necessary because in paraview there are 2 cells per n
     
def writeTimeFunctions():
    functionList=[]
    #functionList.append(Function(functionSpaceTime,timeValuesTime[:],"time"))
    functionList.append(Function(functionSpaceTime,L2timeValuesU[:],"||u||"))
    functionList.append(Function(functionSpaceTime,intTimeValuesU[:],"int u"))
    functionList.append(Function(functionSpaceTime,L2timeValuesGradU[:],"||grad u||"))
    functionList.append(Function(functionSpaceTime,L2timeValuesLaplaceU[:],"||laplace u||"))
    functionList.append(Function(functionSpaceTime,intUUxhoch2ValuesU[:],"int u u_x^2"))
    functionList.append(Function(functionSpaceTime,kdvSpeed[:],"kdv speed"))
    functionList.append(Function(functionSpaceTime,kdvSpeedInParaview[:],"kdv speed (cells/delta t)"))
    File(timeFunctionsFilePath).write(*functionList, time=t)


saveTimeFunctionsArray()
writeTimeFunctions()





### copy script to save it ###
scriptTimeStamp = str(datetime.datetime.now())
scriptTimeStamp = scriptTimeStamp.replace(":","-")
scriptTimeStamp = scriptTimeStamp.replace(" ","_")
scriptCopyPath = output_dir_path + "used_script_simulation_"+scriptTimeStamp+".py"
copy(os.path.realpath(__file__), scriptCopyPath)

infoString += "\n\t"+"scriptCopyPath"+" = \t\t"+str(scriptCopyPath)



### simulating ###

timeStartSolving = datetime.datetime.now()
lastRealTime = timeStartSolving

infoString += "\n"
infoString += "\n\t"+"timeStartSolving"+" = \t\t"+str(timeStartSolving)
infoFile = open(infoFilePath,"a")
infoFile.write(infoString)
infoFile.close()
infoString = ""

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
        
        
    
    
    ##### outputs ###
    if t >= lastWrittenOutput + writeOutputEvery:
        lastWrittenOutput = t
        t_iOutput += 1
        saveTimeFunctionsArray(t_iOutput)
        
        if not writeTimeFunctionsOnlyAtTheEnd:
            writeTimeFunctions()
    
        ### write output mesh functions ###   
        writeMeshFunctions()





    if (time.time()-lastLogOutput)>maxLogOutputsPerSecond:
        print(np.round(t_i/numberOfTimesteps*100,2),"% ( step = ", t_i, " of ", numberOfTimesteps,", time t = ", np.round(t,4),") after ", datetime.datetime.now()-lastRealTime, ", estimated time left ", ((T_end-T_0)/t-1)*(datetime.datetime.now()-timeStartSolving)  )
        lastLogOutput = time.time()
    lastRealTime = datetime.datetime.now()
    
saveTimeFunctionsArray(t_iOutput+1)
writeTimeFunctions()




time_end = datetime.datetime.now()
print("ending at ",time_end)
totalTime = time_end-time_start
print("total time ", totalTime)
infoString += "\n"
infoString += "\n\t"+"time_end"+" = \t\t"+str(time_end)
infoString += "\n\t"+"totalTime"+" = \t\t"+str(totalTime)
infoFile = open(infoFilePath,"a")
infoFile.write(infoString)
infoFile.close()
infoString = ""





