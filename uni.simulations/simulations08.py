from firedrake import *
import numpy as np
import os

### time ###
import datetime

### code source code file ###
from shutil import copy
from ufl.tensors import as_scalar
from ufl.operators import div, grad
from firedrake.variational_solver import LinearVariationalSolver
from networkx.algorithms.centrality.flow_matrix import InverseLaplacian
from firedrake.ufl_expr import TrialFunction
from firedrake.functionspace import FunctionSpace


# using now() to get current time  
time_start = datetime.datetime.now()
print("starting at ",time_start)  






### PARAMETERS ###

# interval lengths
L = 64
L_x = L
L_y = L

# spatial steps
nProL = 1
n_x = L_x*nProL
n_y = L_y*nProL
#n = 64*2
#n_x = n
#n_y = n

# times
numberOfTimestepsPerUnit = 50
T_end = 500

# pde name
# list of available shortNames: nonLin, onlyAdv, advLap, advLap2, advLap2Lap, kuraSiva
pdeShortName = "advLap"

# finite elements
finitEleFamily = "CG"           #https://www.firedrakeproject.org/variational-problems.html#supported-finite-elements
finitEleDegree = 1


# force 0 average after every step? True, False12
forceZeroAverage = False

# kappa in theta_t + < u_adv, grad theta> + kappa*laplace theta + laplace^2 theta = 0
kappa = 1

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

### PARAMETERS END ###







T_0 = 0
numberOfTimesteps = round(numberOfTimestepsPerUnit*(T_end-T_0))
timestep = (T_end-T_0)/numberOfTimesteps

# mesh = UnitSquareMesh(n, n)
#mesh = PeriodicUnitSquareMesh(n,n)
mesh = PeriodicRectangleMesh(n_x,n_y,L_x,L_y)

V = FunctionSpace(mesh, finitEleFamily, finitEleDegree)
V_vec = VectorFunctionSpace(mesh, finitEleFamily, finitEleDegree)
V_out = FunctionSpace(mesh, finitEleFamily, 1)
x, y = SpatialCoordinate(mesh)




### initial data ###
ic_theta = Function(V)

ic_theta.interpolate(ic_scale*sin(ic_freq_x*2*pi*y/L_y))
#ic_theta.interpolate(ic_scale*e**(-ic_freq_x*((x-L_x/2)/L_x)**2-ic_freq_y*((y-L_y/2)/L_y)**2))
#ic_theta.interpolate(-scale*cos(freq_x*2*pi*x/L_x))
#ic_theta.interpolate(-scale*cos(freq_y*2*pi*y/L_y))

### random initial data ###
if randomIC:
    u_0Random = ic_scale*(2*np.random.rand(n_x*n_y)-1)       # 2*... - 1 -> -1 und 1
    ic_theta = Function(V,u_0Random)




#u_adv = project(as_vector([sin(freq_x*2*pi*y/L_y), sin(freq_x*2*pi*x/L_x)]), V)
#u_adv = project(as_vector([adv_scaleX*sin(adv_freqX*2*pi*y/L_y), adv_scaleY*sin(adv_freqY*2*pi*x/L_x)]), V_vec)
#u_adv = project(as_vector([y/L_y,0]), V_vec)
u_adv = project(as_vector([2*(x-L_x/2),-2*(y-L_y/2)]), V_vec)
#u_adv = project(as_vector([1, 1]), V)
if advZero:
    u_adv.assign(0)





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

print("inital values assigned after ",datetime.datetime.now()-time_start)  

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
    + inner(grad(theta), grad(testFunctionA))
    )*dx

if numberTestFunctions == 2:
    F_advLap2Lap = (inner((theta - theta_old)/timestep, testFunctionA)
        + inner(dot(u_adv,grad(theta)), testFunctionA) 
        + kappa*inner(theta_laplace, testFunctionA)
        - inner(grad(theta_laplace), grad(testFunctionA))
        + inner(theta_laplace, testFunctionB)
        + inner(grad(theta), grad(testFunctionB))
        )*dx
    
    F_advLap2 = (inner((theta - theta_old)/timestep, testFunctionA)
        + inner(dot(u_adv,grad(theta)), testFunctionA)
        - inner(grad(theta_laplace), grad(testFunctionA))
        + inner(theta_laplace, testFunctionB)
        + inner(grad(theta), grad(testFunctionB))
        )*dx
        
    F_kuraSiva = (inner((theta - theta_old)/timestep, testFunctionA)
        + 1/2*inner(dot(grad(theta),grad(theta)), testFunctionA) 
        + inner(theta_laplace, testFunctionA)
        - inner(grad(theta_laplace), grad(testFunctionA))
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






###################################
def inverseLaplacian(function):
    # returns lap^-1 f
    # f=function, u = outFunction
    # lap u = f
    # <lap u, v> =<f,v>
    # <-grad u, grad v> = <f,v>
    # <-grad u, grad v> - <f,v> = 0
    # <grad u, grad v> + <f,v> = 0 
    #print("1",function.dat.data)

    outFunction = Function(function.function_space())
    testFunctionInvLap = TestFunction(function.function_space())
    F_invLap = (inner(grad(outFunction),grad(testFunctionInvLap))+dot(function,testFunctionInvLap))*dx
    solve(F_invLap == 0, outFunction)
    return outFunction
###################################
def calcLaplacian(function):
    gradFunction = project(grad(function), V_vec)
    return project(div(gradFunction),V)
###################################
def getZeroAverageOfScalarFunction(function):
    sum = np.sum(function.dat.data)
    function.dat.data[:] = function.dat.data[:]-sum/(n_x*n_y)
    return function
###################################

### batchelor scale ###
### WRONG BATCHELOR SCALE IS L_BAT = (KAPPA / \| \nabla u_adv\|)**(1/2) BUT ONLY FOR adv diff pde (advection heat equation) where kappa laplace theta
if False:
    # force rescale u_adv -> l_bat = 1    (rescales the amplitude)
    forceBatScale = False
    forcedBatScale = 30
    
    if norm(grad(u_adv),"l2") == 0:
        print("advection = 0 -> setting l_bat = 0")
        l_bat = 0
    else:
        if forceBatScale:
            print("l2 of grad u_adv (non rescaled)", str(norm(grad(u_adv),"l2")))
            l_bat = sqrt(kappa/(norm(grad(u_adv),"l2")))
            print("l_bat (non rescaled)", str(l_bat))
            rescaleFactor = kappa/(norm(grad(u_adv),"l2"))/(forcedBatScale**2)
            print("rescale factor ",str(rescaleFactor))
            u_adv.dat.data[:] = rescaleFactor*u_adv.dat.data[:]
        l_bat = sqrt(kappa/(norm(grad(u_adv),"l2")))
    
    print("batchelor scale l_bat=",str(l_bat))
    l_batFunction = Function(V_out)
    l_batFunction.assign(l_bat)



l_dom = (2*sqrt(2)*pi)*(kappa**(-1/2))
print("dominant wavelength (of laplace laplace^2) l_dom=",str(l_dom))







##### outputs #####
output_dir_path = os.path.dirname(os.path.realpath(__file__))


### output mesh functions ####

t = T_0


l_domFunction = Function(V_out)
l_domFunction.assign(l_dom)

gradTheta = Function(V_vec)
gradTheta = project(grad(theta), V_vec)

gradThetaX = Function(V_out)
gradThetaX.dat.data[:] = gradTheta.dat.data[:,0]

gradThetaY = Function(V_out)
gradThetaY.dat.data[:] = gradTheta.dat.data[:,1]

lapInvLapTheta = calcLaplacian(inverseLaplacian(theta))


outfile_theta = File(output_dir_path + "/../data/temp/theta.pvd")
outfile_theta.write(project(theta, V_out, name="theta"),project(theta, V_out, name="theta again"),project(lapInvLapTheta, V_out, name="lap (lap^-1(theta))"),project(theta - lapInvLapTheta, V_out, name="theta - lap (lap^-1(theta))"),project(gradThetaX, V_out, name="d/dx theta"),project(gradThetaY, V_out, name="d/dy theta"), project(l_domFunction, V_out, name="l_dom"), time=t)






### output time functions ###

meshTime = IntervalMesh(numberOfTimesteps+2, T_0, T_end)
VecSpaceTime = VectorFunctionSpace(meshTime, "DG", 0)

timeValuesTime = np.zeros(numberOfTimesteps+2)
timeValuesTime[0] = T_0
TimeFunctionTime = Function(VecSpaceTime,timeValuesTime[:])

L2timeValuesTheta = np.zeros(numberOfTimesteps+2)
L2timeValuesTheta[0] = norm(ic_theta,"l2")
L2normTimeFunctionTheta = Function(VecSpaceTime,L2timeValuesTheta[:])

L2timeValuesddxTheta = np.zeros(numberOfTimesteps+2)
L2timeValuesddxTheta[0] = norm(gradThetaX,"l2")
L2normTimeFunctionddxTheta = Function(VecSpaceTime,L2timeValuesddxTheta[:])

L2timeValuesddyTheta = np.zeros(numberOfTimesteps+2)
L2timeValuesddyTheta[0] = norm(gradThetaY,"l2")
L2normTimeFunctionddyTheta = Function(VecSpaceTime,L2timeValuesddyTheta[:])

outfile_timeFunctions = File(output_dir_path + "/../data/temp/timeFunctions.pvd")
outfile_timeFunctions.write(project(TimeFunctionTime, VecSpaceTime, name="time"),project(L2normTimeFunctionTheta, VecSpaceTime, name="theta L^2"),project(L2normTimeFunctionddxTheta, VecSpaceTime, name="d/dx Theta L^2"),project(L2normTimeFunctionddyTheta, VecSpaceTime, name="d/dy Theta L^2"), time=t)

L2normTimeFunction = Function(VecSpaceTime,L2timeValuesTheta[:])







### copy script to save it ####
copy(os.path.realpath(__file__), output_dir_path + "/../data/temp/0used_script.py")













print(theta.dat.data)
invLapTheta = inverseLaplacian(theta)
tempTheta = calcLaplacian(invLapTheta)
print(tempTheta.dat.data)
print("l2 theta",norm(theta,"l2")," l2 lap lap^-1 theta",norm(tempTheta,"l2")," l2 theta - lap lap^-1 theta",norm(theta-tempTheta,"l2"))
### simulating ###

timeStartSolving = datetime.datetime.now()
lastRealTime = timeStartSolving
t_i=0
while (t < T_end):
    solver.solve()
    t += timestep
    t_i += 1 
    if numberTestFunctions == 1:
        theta_old.assign(theta)
    else:
        w_old.assign(w)
        theta, theta_laplace = w.split()
        
    if forceZeroAverage:
        theta = getZeroAverageOfScalarFunction(theta)
        
    ### write output theta ###
    gradTheta = project(grad(theta), V_vec)
    gradThetaX.dat.data[:] = gradTheta.dat.data[:,0]
    gradThetaY.dat.data[:] = gradTheta.dat.data[:,1]
    lapInvLapTheta = calcLaplacian(inverseLaplacian(theta))
    outfile_theta.write(project(theta, V_out, name="theta"),project(theta, V_out, name="theta again"),project(lapInvLapTheta, V_out, name="lap (lap^-1(theta))"),project(theta - lapInvLapTheta, V_out, name="theta - lap (lap^-1(theta))"),project(gradThetaX, V_out, name="d/dx theta"),project(gradThetaY, V_out, name="d/dy theta"), project(l_domFunction, V_out, name="l_dom"), time=t)
    
    
    ### write output time functions ###
    timeValuesTime[t_i] = t
    TimeFunctionTime = Function(VecSpaceTime,timeValuesTime[:])
    
    L2timeValuesTheta[t_i] = norm(theta,"l2")
    L2normTimeFunctionTheta = Function(VecSpaceTime,L2timeValuesTheta[:])
    
    L2timeValuesddxTheta[t_i] = norm(gradThetaX,"l2")
    L2normTimeFunctionddxTheta = Function(VecSpaceTime,L2timeValuesddxTheta[:])
    
    L2timeValuesddyTheta[t_i] = norm(gradThetaY,"l2")
    L2normTimeFunctionddyTheta = Function(VecSpaceTime,L2timeValuesddyTheta[:])
    
    outfile_timeFunctions.write(project(TimeFunctionTime, VecSpaceTime, name="time"),project(L2normTimeFunctionTheta, VecSpaceTime, name="theta L^2"),project(L2normTimeFunctionddxTheta, VecSpaceTime, name="d/dx Theta L^2"),project(L2normTimeFunctionddyTheta, VecSpaceTime, name="d/dy Theta L^2"), time=t)
    

    print(np.round(t_i/numberOfTimesteps*100,2),"% ( step = ", t_i, " von ", numberOfTimesteps,", time t = ", np.round(t,4),") after ", datetime.datetime.now()-lastRealTime, ", estimated time left ", ((T_end-T_0)/t-1)*(datetime.datetime.now()-timeStartSolving)  )
    lastRealTime = datetime.datetime.now()

time_end = datetime.datetime.now()
print("ending at ",time_end)
print("total time ", time_end-time_start)
logFile = open(output_dir_path + "/../data/temp/0log.txt","a") 
parametersString += ["\nending at ",str(time_end),"\ntotal time in hh:min:sec = ",str(time_end-time_start)]
logFile.writelines(parametersString)
logFile.close()




##### old #####

### log file ###
if False:
    logFile = open(output_dir_path + "/../data/temp/0log.txt","w") 
    parametersString = ["# interval lengths","\nL_x = ",str(L_x),"\nL_y = ",str(L_y)]
    parametersString += ["\n\n# spatial steps","\nn_x = ",str(n_x),"\nn_y = ",str(n_y)]
    parametersString += ["\n\n# time steps","\nnumberOfTimesteps = ",str(numberOfTimesteps),"\nT_0 = ",str(T_0),"\nT_end = ",str(T_end)]
    parametersString += ["\n\n# pde","\npdeShortName = ",str(pdeShortName)]
    parametersString += ["\n\n# finite elements","\nfamily = ",finitEleFamily,"\ndegree = ",str(finitEleDegree)]
    parametersString += ["\n\n# force 0 average after every step?","\nforceZeroAverage = ",str(forceZeroAverage)]
    parametersString += ["\n\n# batchelor scale","\nl_bat = ",str(l_bat)]
    parametersString += ["\n\n# time for simulation","\nstarting at ",str(time_start)]
    logFile.writelines(parametersString)
    logFile.close()



