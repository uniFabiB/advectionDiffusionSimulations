from firedrake import *
import numpy as np
import os

### time ###
import datetime

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

# interval lengths
L = 50
L_x = L
L_y = L

# spatial steps
n = 16*4
n_x = n
n_y = n

# times
numberOfTimesteps = 500
T_0 = 0
T_end = 50.0

timestep = (T_end-T_0)/numberOfTimesteps

# mesh = UnitSquareMesh(n, n)
#mesh = PeriodicUnitSquareMesh(n,n)
mesh = PeriodicRectangleMesh(n,n,L_x,L_y)

V = VectorFunctionSpace(mesh, "CG", 1)
V_out = VectorFunctionSpace(mesh, "CG", 1)


u = Function(V)
u_ = Function(V)

#################################
x, y = SpatialCoordinate(mesh)
scale = 1
freq = 1
freq_x = 1
freq_y = 3
ic_u = project(as_vector([scale*sin(freq_x*2*pi*x/L_x), scale*sin(freq_x*2*pi*y/L_x)]), V)
omega = project(as_vector([1, -1]), V)
u.assign(ic_u)
u_.assign(ic_u)


print("inital values assigned after ",datetime.datetime.now()-time_start)  

#bc_u = DirichletBC(W.sub(0), as_vector([scale*sin(freq*pi*x),0]), "on_boundary")
#bc_v = DirichletBC(W.sub(1), as_vector([-(freq*pi)**2*scale*sin(freq*pi*x),0]), "on_boundary")

#################################

u_test = TestFunction(V)

#################################

F_onlyNonLin = (inner((u - u_)/timestep, u_test)
    + inner(dot(u,grad(u)), u_test) 
    )*dx
    
F_advection = (inner((u - u_)/timestep, u_test)
    + inner(dot(omega,grad(u)), u_test)
    )*dx
    
F_advectionLaplace = (inner((u - u_)/timestep, u_test)
    + inner(dot(omega,grad(u)), u_test) 
    + inner(grad(u), grad(u_test))
    )*dx
    
F_advectionLaplace = (inner((u - u_)/timestep, u_test)
    + inner(dot(omega,grad(u)), u_test) 
    + inner(grad(u), grad(u_test))
    )*dx

F = F_advection
#################################

#################################
#F = (inner((u - u_)/timestep, u_test)
#    + inner(dot(u,nabla_grad(u)), u_test)
#    #- nu*inner(v, u_test)
#    #- mu*inner(grad(v), grad(u_test))
#    + inner(v, v_test)
#    + inner(grad(u), grad(v_test)))*dx
#################################


# problem = NonlinearVariationalProblem(F, w, bcs=[bc_u, bc_v])
problem = NonlinearVariationalProblem(F, u)

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
outfile_u1 = File(output_dir_path + "/../data/kse_u1.pvd")
outfile_u1.write(project(u, V_out, name="Velocity1"), time=t)

#outfile_v = File(output_dir_path + "/../data/kse_v.pvd")
#outfile_v.write(project(v, V_out, name="Velocity_xx2"))

outfile_u2 = File(output_dir_path + "/../data/kse_u2.pvd")
outfile_u2.write(project(u, V_out, name="Velocity2"), time=t)

outfile_u3 = File(output_dir_path + "/../data/kse_u3.pvd")
outfile_u3.write(project(u, V_out, name="Velocity3"), time=t)



while (t < T_end):
    solver.solve()
    u_.assign(u)
    t += timestep
    outfile_u1.write(project(u, V_out, name="Velocity1"), time=t)
    outfile_u2.write(project(u, V_out, name="Velocity2"), time=t)
    outfile_u3.write(project(u, V_out, name="Velocity3"), time=t)
    print(np.round(t/(T_end-T_0)*100,2),"% ( time t = ", t, " von ", np.round(T_end-T_0),")")


time_end = datetime.datetime.now()
print("ending at ",time_end)
print("total time ", time_end-time_start)  

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




