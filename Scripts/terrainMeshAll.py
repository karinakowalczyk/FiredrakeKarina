import matplotlib.pyplot as plt
from firedrake import *
#import petsc4py.PETSc as PETSc
#PETSc.Sys.popErrorHandler()

##############################################################################################
#####################                    Piola                      ##########################
##############################################################################################


##############     define mesh ######################################################################

m = IntervalMesh(10,1)
mesh = ExtrudedMesh(m, 5, extrusion_type='uniform')

Vc = mesh.coordinates.function_space()
x, y = SpatialCoordinate(mesh)
#f_mesh = Function(Vc).interpolate(as_vector([x, y + ( 0.25 * x**4 -x**3 + x**2) * (1-y) ] ) )
f_mesh = Function(Vc).interpolate(as_vector([x,y - exp(-x**2/2)*((y-0.5)**2 -0.25)] ) )
mesh.coordinates.assign(f_mesh)

xs = [mesh.coordinates.dat.data[i][0] for i in range(0,66)]
ys = [mesh.coordinates.dat.data[i][1] for i in range(0,66)]

plt.scatter(xs, ys)

########################### define function spaces #####################################
CG_1 = FiniteElement("CG", interval, 1)
DG_0 = FiniteElement("DG", interval, 0)
P1P0 = TensorProductElement(CG_1, DG_0)
RT_horiz = HDivElement(P1P0)
P0P1 = TensorProductElement(DG_0, CG_1)
RT_vert = HDivElement(P0P1)
element = RT_horiz + RT_vert

#Sigma = FunctionSpace(mesh, "RTCF", 1)
Sigma = FunctionSpace(mesh, element)
VD = FunctionSpace(mesh, "DQ", 0)

W = Sigma * VD

############# define problem ########################################################
sigma, u = TrialFunctions(W)
tau, v = TestFunctions(W)

#boundary conditions sigma dot n = 0 everywhere:
bc0 = DirichletBC(W.sub(0), as_vector([0.0, 0.0]), "bottom")
bc1 = DirichletBC(W.sub(0), as_vector([0.0, 0.0]), "top")
bc2 = DirichletBC(W.sub(0), as_vector([0.0, 0.0]), 1)
bc3 = DirichletBC(W.sub(0), as_vector([0.0, 0.0]), 2)

#force term
x, y = SpatialCoordinate(mesh)
f = 10*exp(-100*((x - 1)**2 + (y - 0.5)**2))

#set up variational problem
a = dot(sigma, tau)*dx  + div(tau)*u*dx + div(sigma)*v*dx + v*u*dx
L = -f*v*dx

#will be the solution
wh = Function(W)

#solve

parameters = {"ksp_type":"gmres", "ksp_monitor":None, "pc_type":"lu", "mat_type":"aij","pc_factor_mat_solver_type":"mumps" }

solve(a == L, wh, bcs=[bc0, bc1, bc2, bc3], solver_parameters=parameters)

sigmah, uh = wh.split()

#output/plot solution
file0 = File("../Results/Piola.pvd")
file0.write(sigmah, uh)

fig, axes = plt.subplots()
quiver(sigmah, axes=axes)
axes.set_aspect("equal")
axes.set_title("$\sigma$")
plt.show()


#############           hybridized with SCPC         ####################################


Sigmahat = FunctionSpace(mesh, BrokenElement(Sigma.ufl_element()))
V = FunctionSpace(mesh, VD.ufl_element())
T = FunctionSpace(mesh, FiniteElement("HDiv Trace", mesh.ufl_cell(), degree=0))
W_hybrid = Sigmahat * V * T

n = FacetNormal(mesh)
sigmahat, uhat, lambdar = TrialFunctions(W_hybrid)
tauhat, vhat, gammar = TestFunctions(W_hybrid)

wh = Function(W_hybrid)

a_hybrid = (inner(sigmahat, tauhat) * dx + div(tauhat) * uhat * dx - div(sigmahat) * vhat * dx + vhat * uhat * dx
            + inner(tauhat, n) * lambdar * (ds_b + ds_t + ds_v)
            + inner(sigmahat, n) * gammar * (ds_b + ds_t + ds_v)
            + jump(tauhat, n=n) * lambdar('+') * (dS_h + dS_v)
            + jump(sigmahat, n=n) * gammar('+') * (dS_h + dS_v))

L = f * vhat * dx

scpc_parameters = {"ksp_type": "preonly", "pc_type": "lu"}

solve(a_hybrid == L, wh, solver_parameters = {"ksp_type": "gmres","mat_type":"matfree",
                                              "pc_type":"python", "pc_python_type":"firedrake.SCPC",
                                              "condensed_field":scpc_parameters,
                                              "pc_sc_eliminate_fields":"0,1"})

sigmah, uh, lamdah = wh.split()

uh_Piola = Function(V)
uh_Piola = uh
sigmah_Piola = Function(Sigmahat)
sigmah_Piola = sigmah

V_plot = VectorFunctionSpace(mesh, "DG", 1)
sigmah_plot = Function(V_plot).project(sigmah)

file2 = File("../Results/PiolaSCPC.pvd")
file2.write(sigmah_plot, uh)

quiver(sigmah, axes=axes)
axes.set_aspect("equal")
axes.set_title("$\sigma$")

#####################     hybridized with HybridizationPC      #################################################

hybrid_parameters = {'ksp_type': 'preonly',
                     'mat_type': 'matfree',
                     'pc_type': 'python',
                     'pc_python_type': 'firedrake.HybridizationPC',
                     # Solver for the trace system
                     'hybridization': {'ksp_type': 'gmres',
                                       'pc_type': 'gamg',
                                       'pc_gamg_sym_graph': True,
                                       'ksp_rtol': 1e-7,
                                       'mg_levels': {'ksp_type': 'richardson',
                                                     'ksp_max_it': 5,
                                                     'pc_type': 'bjacobi',
                                                     'sub_pc_type': 'ilu'}}}

sigma, u = TrialFunctions(W)
tau, v = TestFunctions(W)

wh = Function(W)

L = -f*v*dx

problem = LinearVariationalProblem(a, L, wh, bcs=[bc0,bc1, bc2, bc3])

solver = LinearVariationalSolver(problem, solver_parameters=hybrid_parameters)

solver.solve()

sigmah,  uh = wh.split()

file2 = File("../Results/PiolaHybridizationPC.pvd")
file2.write(sigmah, uh)

fig, axes = plt.subplots()

quiver(sigmah, axes=axes)
axes.set_aspect("equal")
axes.set_title("$\sigma$")



#######################################################################################################################
##################################       Identity         #############################################################
#######################################################################################################################



#################### define Function Spaces ###########################################################################

element = FiniteElement("RTCF", cell="quadrilateral", degree=1)
element._mapping = 'identity'
Sigma = FunctionSpace(mesh, element)
V = FunctionSpace(mesh, "DQ", 0)

Sigmahat = FunctionSpace(mesh, BrokenElement(Sigma.ufl_element()))  # do I need broken element here??
V = FunctionSpace(mesh, V.ufl_element())
T = FunctionSpace(mesh, FiniteElement("HDiv Trace", mesh.ufl_cell(), degree=0))
W_hybrid = Sigmahat * V * T

n = FacetNormal(mesh)

sigmahat, uhat, lambdar = TrialFunctions(W_hybrid)
tauhat, vhat, gammar = TestFunctions(W_hybrid)

wh = Function(W_hybrid)

f = 10 * exp(-100 * ((x - 1) ** 2 + (y - 0.5) ** 2))

a_hybrid = (inner(sigmahat, tauhat) * dx + div(tauhat) * uhat * dx + div(sigmahat) * vhat * dx + vhat * uhat * dx
            + inner(tauhat, n) * lambdar * (ds_b + ds_t + ds_v)
            + inner(sigmahat, n) * gammar * (ds_b + ds_t + ds_v)
            + jump(tauhat, n=n) * lambdar('+') * (dS_h + dS_v)
            + jump(sigmahat, n=n) * gammar('+') * (dS_h + dS_v))

L = -f * vhat * dx

# will be the solution
wh = Function(W_hybrid)

# solve

parameters = {"ksp_type": "gmres", "ksp_monitor": None,
              "pc_type": "lu", "mat_type": "aij",
              "pc_factor_mat_solver_type": "mumps"}

solve(a_hybrid == L, wh, bcs = [], solver_parameters=parameters)

sigmah, uh, lambdah = wh.split()

#output/plot solution
file0 = File("../Results/Identity.pvd")
#can't write sigmah to file???
file0.write(sigmah,uh)
fig, axes = plt.subplots()
quiver(sigmah, axes=axes)
axes.set_aspect("equal")
axes.set_title("$\sigma$")

#######################################  hybridized with SCPC   #####################################################


scpc_parameters = {"ksp_type":"preonly", "pc_type":"lu"}

solve(a_hybrid == L, wh, solver_parameters = {"ksp_type": "gmres","mat_type":"matfree",
                                              "pc_type":"python", "pc_python_type":"firedrake.SCPC",
                                              "condensed_field":scpc_parameters,
                                              "pc_sc_eliminate_fields":"0,1"})

sigmah, uh, lamdah = wh.split()

uh_Identity = Function(V)
uh_Identity = uh
sigmah_Identity = Function(Sigmahat)
sigmah_Identity = sigmah

V_plot = VectorFunctionSpace(mesh, "DG", 1)
sigmah_plot = Function(V_plot).interpolate(sigmah)

file2 = File("../Results/IdentitySCPC.pvd")
file2.write(sigmah_plot, uh)

fig, axes = plt.subplots()

quiver(sigmah, axes=axes)
axes.set_aspect("equal")
axes.set_title("$\sigma$")


###################################### HybridizationPC ###############################################################
W = Sigma * V

sigma, u = TrialFunctions(W)
tau, v = TestFunctions(W)



#how do I implent BC on deformed mesh????

bc0 = DirichletBC(W.sub(0), 0, "bottom") ##in undeformed grid these are the normal components
bc1 = DirichletBC(W.sub(0), 0, "top")
bc2 = DirichletBC(W.sub(0), 0, 1)
bc3 = DirichletBC(W.sub(0), 0,2)

#force term
x, y = SpatialCoordinate(mesh)
f = 10*exp(-100*((x - 1)**2 + (y - 0.5)**2))

#set up variational problem
a = dot(sigma, tau)*dx  + div(tau)*u*dx + div(sigma)*v*dx + u*v*dx
L = -f*v*dx

#will be the solution
wh = Function(W)


########### solve ###################################################

problem = LinearVariationalProblem(a, L, wh, bcs=[bc0,bc1, bc2, bc3])

solver = LinearVariationalSolver(problem, solver_parameters=hybrid_parameters)

solver.solve()
sigmah, uh = wh.split()

#output/plot solution
file0 = File("../Results/IdentityHybridizationPC.pvd")
#can't write sigmah to file???
file0.write(sigmah,uh)
fig, axes = plt.subplots()
quiver(sigmah, axes=axes)
axes.set_aspect("equal")
axes.set_title("$\sigma$")


###################################################################################################################
############################ compare Piola and Identity Transformations ###########################################

#differenceU = Function(V)
differenceU = uh_Piola - uh_Identity

V_plot = FunctionSpace(mesh, "DG", 1)
diffU_plot = Function(V_plot).interpolate(differenceU)
File = File("../Results/DifferencePiolaIdentity.pvd")

Sigma_plot = VectorFunctionSpace(mesh, "DG", 1)
diffSigma_plot = Function(Sigma_plot).project(sigmah_Piola - sigmah_Identity)


File.write(diffU_plot, diffSigma_plot)

diffU = norm(differenceU, norm_type = "L2")
print('difference of u:')
print(diffU)
