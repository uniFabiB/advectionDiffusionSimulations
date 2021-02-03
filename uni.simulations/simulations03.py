from firedrake import *
import numpy as np
import os

### time ###
import datetime

### code source code file ###
from shutil import copy
from firedrake.utility_meshes import IntervalMesh


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
L = 100
L_x = L
L_y = L

# spatial steps
n = 16
n_x = n
n_y = n

# times
numberOfTimesteps = 100
T_end = 10

# pde name
# list of available shortNames: nonLin, onlyAdv, advLap, advLap2, advLap2Lap
pdeShortName = "advLap"

# finite elements
finitEleFamily = "CG"           #https://www.firedrakeproject.org/variational-problems.html#supported-finite-elements
finitEleDegree = 1
### PARAMETERS END ###


print("todo kura siva rot free")




T_0 = 0
timestep = (T_end-T_0)/numberOfTimesteps

# mesh = UnitSquareMesh(n, n)
#mesh = PeriodicUnitSquareMesh(n,n)
mesh = PeriodicRectangleMesh(n,n,L_x,L_y)

V = VectorFunctionSpace(mesh, finitEleFamily, finitEleDegree)
V_out = VectorFunctionSpace(mesh, finitEleFamily, 1)
x, y = SpatialCoordinate(mesh)



scale = 1
freq = 1
freq_x = 1
freq_y = 3
ic_theta = project(as_vector([scale*sin(freq_x*2*pi*x/L_x), scale*sin(freq_x*2*pi*y/L_x)]), V)
u_adv = project(as_vector([3*sin(freq_x*2*pi*y/L_y), 3*sin(freq_x*2*pi*x/L_x)]), V)
#u_adv = project(as_vector([1, 1]), V)
#u_adv.assign(0)


u_0Random1 = 2*np.random.rand(n_x*n_y)-1       # zwischen 0 und 1 -> -1 und 1
u_0Random2 = 2*np.random.rand(n_x*n_y)-1       # zwischen 0 und 1 -> -1 und 1
#ic_theta = Function(V,[u_0Random1,u_0Random2])

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

F_nonLin = (inner((theta - theta_old)/timestep, testFunctionA)
    + inner(dot(theta,grad(theta)), testFunctionA) 
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
        + inner(dot(theta,grad(theta)), testFunctionA) 
        + inner(theta_laplace, testFunctionA)
        - inner(grad(theta_laplace), grad(testFunctionA))
        + inner(theta_laplace, testFunctionB)
        + inner(grad(theta), grad(testFunctionB))
        )*dx
    
    F_advLap2 = (inner((theta - theta_old)/timestep, testFunctionA)
        + inner(dot(theta,grad(theta)), testFunctionA)
        - inner(grad(theta_laplace), grad(testFunctionA))
        + inner(theta_laplace, testFunctionB)
        + inner(grad(theta), grad(testFunctionB))
        )*dx
        
    F_kuraSiva = (inner((theta - theta_old)/timestep, testFunctionA)
        + inner(dot(theta,grad(theta)), testFunctionA) 
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

#################################

#################################
#F = (inner((u - u_)/timestep, u_test)
#    + inner(dot(u,nabla_grad(u)), u_test)
#    #- nu*inner(v, u_test)
#    #- mu*inner(grad(v), grad(u_test))
#    + inner(v, v_test)
#    + inner(grad(u), grad(v_test)))*dx
#################################

def getL2Norms(array):
    #print(array)
    ## calculates the L² norms of the array
    ## L² in x in the 1st variable 
    ## L² in y in the 2nd variable
    ## total L² in 3 variable
    def getL2NormOfVector(vector):
        #print(vector)
        #print(sqrt(vector.dot(vector)))
        return sqrt(vector.dot(vector))
    L2x = getL2NormOfVector(array[:,0])/n_x
    L2y = getL2NormOfVector(array[:,1])/n_y
    Lges = sqrt(L2x*L2x+L2y*L2y)
    #print([L2x, L2y, Lges])
    return [L2x, L2y, Lges]

###################################

# problem = NonlinearVariationalProblem(F, w, bcs=[bc_u, bc_v])
if numberTestFunctions == 1:
    problem = NonlinearVariationalProblem(F, theta)
else:
    problem = NonlinearVariationalProblem(F, w)
    
    

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

t = T_0
t_function = Function(V)
t_function.assign(t)

outfile_theta = File(output_dir_path + "/../data/temp/kse_theta.pvd")
L2norm = getL2Norms(ic_theta.dat.data)
outfile_theta.write(project(t_function, V_out, name="time"),project(project(as_vector([L2norm[0], L2norm[1]]), V), V_out, name="L2[x,y]"),project(project(as_vector([L2norm[2], L2norm[2]]), V), V_out, name="L2_tot"),project(theta, V_out, name="theta1"),project(theta, V_out, name="theta2"),project(theta, V_out, name="theta3"), time=t, a=3)

#outfile_v = File(output_dir_path + "/../data/kse_v.pvd")
#outfile_v.write(project(v, V_out, name="Velocity_xx2"))

#outfile_u2 = File(output_dir_path + "/../data/temp/kse_theta2.pvd")
#outfile_u2.write(project(theta, V_out, name="theta2"), time=t)

#outfile_u3 = File(output_dir_path + "/../data/temp/kse_theta3.pvd")
#outfile_u3.write(project(theta, V_out, name="theta3"), time=t)

#outfile_u4 = File(output_dir_path + "/../data/temp/kse_theta4.pvd")
#outfile_u4.write(project(theta, V_out, name="theta4"), time=t)


logFile = open(output_dir_path + "/../data/temp/0log.txt","w") 
parametersString = ["# interval lengths","\nL_x = ",str(L_x),"\nL_y = ",str(L_y)]
parametersString += ["\n\n# spatial steps","\nn_x = ",str(n_x),"\nn_y = ",str(n_y)]
parametersString += ["\n\n# time steps","\nnumberOfTimesteps = ",str(numberOfTimesteps),"\nT_0 = ",str(T_0),"\nT_end = ",str(T_end)]
parametersString += ["\n\n# pde","\npdeShortName = ",str(pdeShortName)]
parametersString += ["\n\n# finite elements","\nfamily = ",finitEleFamily,"\ndegree = ",str(finitEleDegree)]
logFile.writelines(parametersString)
logFile.close()

copy(os.path.realpath(__file__), output_dir_path + "/../data/temp/0used_script.py")


logFile = open(output_dir_path + "/../data/temp/0log.txt","a") 
parametersString += ["\n\n# time for simulation","\nstarting at ",str(time_start)]
logFile.writelines(parametersString)
logFile.close()



timeStartSolving = datetime.datetime.now()
lastRealTime = timeStartSolving
while (t < T_end):
    solver.solve()
    t += timestep
    if numberTestFunctions == 1:
        theta_old.assign(theta)
    else:
        w_old.assign(w)
        theta, theta_laplace = w.split()
    #print(L2Norms(theta.dat.data),"\n")
    t_function.assign(t)
    L2norm = getL2Norms(theta.dat.data)
    outfile_theta.write(project(t_function, V_out, name="time"),project(project(as_vector([L2norm[0], L2norm[1]]), V), V_out, name="L2[x,y]"),project(project(as_vector([L2norm[2], L2norm[2]]), V), V_out, name="L2_tot"),project(theta, V_out, name="theta1"),project(theta, V_out, name="theta2"),project(theta, V_out, name="theta3"), time=t)
    #outfile_u4.write(project(theta, V_out, name="theta4"), time=t)
    print(np.round(t/(T_end-T_0)*100,2),"% ( time t = ", np.round(t,4), " von ", np.round(T_end-T_0),") after ", datetime.datetime.now()-lastRealTime, ", estimated time left ", ((T_end-T_0)/t-1)*(datetime.datetime.now()-timeStartSolving)  )
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




