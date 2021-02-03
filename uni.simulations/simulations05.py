from firedrake import *
import numpy as np
import os

### time ###
import datetime

### code source code file ###
from shutil import copy
from ufl.tensors import as_scalar


### description ###
### here only 1 test function -> up to order 2 variables ###
### u = u solution to pde ###
### v = laplace u solution to pde ###
### u_ = old u (last timestep) ###
### v_ = old v (last timestep) ###
### u_test = testfunction for u ###
### v_test = testfunction for v ###


# using now() to get current time  
time_start = datetime.datetime.now()
print("starting at ",time_start)  






### PARAMETERS ###

# interval lengths
L = 64
L_x = L
L_y = L

# spatial steps
n = 64
n_x = n
n_y = n

# times
numberOfTimestepsPerUnit = 2
T_end = 250

# pde name
# list of available shortNames: nonLin, onlyAdv, advLap, advLap2, advLap2Lap, kuraSiva
pdeShortName = "advLap2Lap"

# finite elements
finitEleFamily = "CG"           #https://www.firedrakeproject.org/variational-problems.html#supported-finite-elements
finitEleDegree = 1


# force 0 average after every step? True, False
forceZeroAverage = False


# kappa in theta_t + < u_adv, grad theta> + kappa*laplace theta + laplace^2 theta = 0
kappa = 1

### PARAMETERS END ###






T_0 = 0
numberOfTimesteps = numberOfTimestepsPerUnit*(T_end-T_0)
timestep = (T_end-T_0)/numberOfTimesteps

# mesh = UnitSquareMesh(n, n)
#mesh = PeriodicUnitSquareMesh(n,n)
mesh = PeriodicRectangleMesh(n_x,n_y,L_x,L_y)

V = FunctionSpace(mesh, finitEleFamily, finitEleDegree)
V_vec = VectorFunctionSpace(mesh, finitEleFamily, finitEleDegree)
V_out = FunctionSpace(mesh, finitEleFamily, 1)
x, y = SpatialCoordinate(mesh)




### initial data ###
ic_scale = 1
ic_freq = 1
ic_freq_x = ic_freq
ic_freq_y = ic_freq
ic_theta = Function(V)

ic_theta.interpolate(ic_scale*sin(ic_freq_y*2*pi*y/L_y))
#ic_theta.interpolate(-scale*cos(freq_x*2*pi*x/L_x))
#ic_theta.interpolate(-scale*cos(freq_y*2*pi*y/L_y))

### random initial data ###
if False:
    u_0Random = ic_scale*(2*np.random.rand(n_x*n_y)-1)       # 2*... - 1 -> -1 und 1
    ic_theta = Function(V,u_0Random)





### advection ###
adv_scale = 1
adv_scaleX = adv_scale
adv_scaleY = adv_scale
adv_freq = 1
adv_freqX = adv_freq
adv_freqY = adv_freq


#u_adv = project(as_vector([sin(freq_x*2*pi*y/L_y), sin(freq_x*2*pi*x/L_x)]), V)
u_adv = project(as_vector([adv_scaleX*sin(adv_freqX*2*pi*y/L_y), adv_scaleY*sin(adv_freqY*2*pi*x/L_x)]), V_vec)
#u_adv = project(as_vector([1, 1]), V)
#u_adv.assign(0)





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

solver = NonlinearVariationalSolver(problem,
                                    solver_parameters=sp_it)


output_dir_path = os.path.dirname(os.path.realpath(__file__))



### output mesh functions ####

t = T_0
t_function = Function(V)
t_function.assign(t)

gradTheta = Function(V_vec)
gradThetaX = Function(V_out)
gradThetaY = Function(V_out)
gradTheta = project(grad(theta), V_vec)
gradThetaX.dat.data[:] = gradTheta.dat.data[:,0]
gradThetaY.dat.data[:] = gradTheta.dat.data[:,1]

    
outfile_theta = File(output_dir_path + "/../data/temp/theta.pvd")





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



### batchelor scale ###
l2GradU_adv = norm(grad(u_adv),"l2")
#GradU_advX = Function(V_vec)
#GradU_advX.assign(grad(u_adv))
#L2ofGradU_adv = sqrt(getActualL2NormOfScalar(GradU_adv)**2+getActualL2NormOfScalar(GradU_adv)**2)
print("l2 of grad u_adv firedrake", str(l2GradU_adv))
#print("l2 of grad u_adv firedrake", str(l2GradU_adv))
#print("sqrt(h1^2 - l2^2) of grad u_adv firedrake", str(sqrt(norm(u_adv,"h1")**2-norm(u_adv,"l2")**2)))

### batchelor scale ###
l_bat = sqrt(kappa/(norm(grad(u_adv),"l2")))
print("batchelor scale l_bat=",str(l_bat)) 




### log file ###
logFile = open(output_dir_path + "/../data/temp/0log.txt","w") 
parametersString = ["# interval lengths","\nL_x = ",str(L_x),"\nL_y = ",str(L_y)]
parametersString += ["\n\n# spatial steps","\nn_x = ",str(n_x),"\nn_y = ",str(n_y)]
parametersString += ["\n\n# time steps","\nnumberOfTimesteps = ",str(numberOfTimesteps),"\nT_0 = ",str(T_0),"\nT_end = ",str(T_end)]
parametersString += ["\n\n# pde","\npdeShortName = ",str(pdeShortName)]
parametersString += ["\n\n# finite elements","\nfamily = ",finitEleFamily,"\ndegree = ",str(finitEleDegree)]
parametersString += ["\n\n# force 0 average after every step?","\nforceZeroAverage = ",str(forceZeroAverage)]
parametersString += ["\n\n# batchelor scale","\nl_bat = ",str(l_bat)]
logFile.writelines(parametersString)
logFile.close()
### copy script to save it ####
copy(os.path.realpath(__file__), output_dir_path + "/../data/temp/0used_script.py")




logFile = open(output_dir_path + "/../data/temp/0log.txt","a") 
parametersString += ["\n\n# time for simulation","\nstarting at ",str(time_start)]
logFile.writelines(parametersString)
logFile.close()





### output parameters ###
meshParameter = IntervalMesh(2, 0, 1)
VecParameter = FunctionSpace(meshParameter, "DG", 0)

outfile_parameter = File(output_dir_path + "/../data/temp/parameter.pvd")
outfile_parameter.write(project(Function(VecParameter,(l_bat,l_bat)), VecParameter,name="l_bat"),time=0)



#################################
def getZeroAverageOfScalarFunction(function):
    sum = np.sum(function.dat.data)
    function.dat.data[:] = function.dat.data[:]-sum/(n_x*n_y)
    return function
###################################




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
        
        
    gradTheta = project(grad(theta), V_vec)
    gradThetaX.dat.data[:] = gradTheta.dat.data[:,0]
    gradThetaY.dat.data[:] = gradTheta.dat.data[:,1]
    
    outfile_theta.write(project(theta, V_out, name="theta"),project(gradThetaX, V_out, name="d/dx theta"),project(gradThetaY, V_out, name="d/dy theta"),project(t_function, V_out, name="time"), time=t)
    #outfile_theta.write(project(t_function, V_out, name="time"),project(tempScalarFunction, V_out, name="L^2_spAvg"),project(theta, V_out, name="theta"), time=t)
    
    timeValuesTime[t_i] = t
    TimeFunctionTime = Function(VecSpaceTime,timeValuesTime[:])
    
    L2timeValuesTheta[t_i] = norm(theta,"l2")
    L2normTimeFunctionTheta = Function(VecSpaceTime,L2timeValuesTheta[:])
    
    L2timeValuesddxTheta[t_i] = norm(gradThetaX,"l2")
    L2normTimeFunctionddxTheta = Function(VecSpaceTime,L2timeValuesddxTheta[:])
    
    L2timeValuesddyTheta[t_i] = norm(gradThetaY,"l2")
    L2normTimeFunctionddyTheta = Function(VecSpaceTime,L2timeValuesddyTheta[:])
    
    outfile_timeFunctions.write(project(TimeFunctionTime, VecSpaceTime, name="time"),project(L2normTimeFunctionTheta, VecSpaceTime, name="theta L^2"),project(L2normTimeFunctionddxTheta, VecSpaceTime, name="d/dx Theta L^2"),project(L2normTimeFunctionddyTheta, VecSpaceTime, name="d/dy Theta L^2"), time=t)
    

    #outfile_u4.write(project(theta, V_out, name="theta4"), time=t)
    print(np.round(t_i/numberOfTimesteps*100,2),"% ( step = ", t_i, " von ", numberOfTimesteps,", time t = ", np.round(t,4),") after ", datetime.datetime.now()-lastRealTime, ", estimated time left ", ((T_end-T_0)/t-1)*(datetime.datetime.now()-timeStartSolving)  )
    lastRealTime = datetime.datetime.now()

time_end = datetime.datetime.now()
print("ending at ",time_end)
print("total time ", time_end-time_start)
logFile = open(output_dir_path + "/../data/temp/0log.txt","a") 
parametersString += ["\nending at ",str(time_end),"\ntotal time in hh:min:sec = ",str(time_end-time_start)]
logFile.writelines(parametersString)
logFile.close()

    
###############################################
#t_i = 0

#xsteps = len(u.vector())

#u_OutputArray = np.empty((xsteps,gesTimeSteps))
#
#while (t_i < gesTimeSteps):
#    u_OutputArray[:,t_i] = w.split()[0].vector()[:]
#    t_i += 1
#    solver.solve()
#    w_.assign(w)
#    u, v = w.split()
#    t = t_i*timeDelta
#    outfile_u.write(project(u, V_out, name="u"), time=t)
#    outfile_v.write(project(v, V_out, name="u_xx"), time=t)
##    outfile_u.write(u, time=time)
##    outfile_v.write(v, time=time)
#    print(t_i/gesTimeSteps, "time step t = ", t, " ....done")
#
#
#np.save(output_dir_path + "/../data/kuraSiva/kse_u.npy", u_OutputArray) 

##### old #####
### use firedrake.norm ###
#def getAvgL2NormOfScalar(function):
#    #print(function)
#    #print(sqrt(function.dot(vector)))
#    print("WARNING, get norm is depricated use firedrake norm")
#    return sqrt(function.dot(function)/(n_x*n_y))
#
#def getActualL2NormOfScalar(function):
#    return sqrt(L_x*L_y)*getAvgL2NormOfScalar(function)
#
#
#
#def getAvgL2NormsOfVector(array):
#    #print(array)
#    ## calculates the L² norms of the array
#    ## L² in x in the 1st variable 
#    ## L² in y in the 2nd variable
#    ## total L² in 3 variable
#    L2x = getL2NormOfVector(array[:,0])
#    L2y = getL2NormOfVector(array[:,1])
#    Lges = sqrt(L2x*L2x+L2y*L2y)
#    #print([L2x, L2y, Lges])
#    return [L2x, L2y, Lges]
##### end of old #####



