import matplotlib.pyplot as plt
from firedrake import *
import petsc4py.PETSc as PETSc
PETSc.Sys.popErrorHandler()


##############define mesh ######################################################################

m = IntervalMesh(10,2)
mesh = ExtrudedMesh(m, 5, extrusion_type='uniform')

Vc = mesh.coordinates.function_space()
x, y = SpatialCoordinate(mesh)
f_mesh = Function(Vc).interpolate(as_vector([x, y + ( 0.25 * x**4 -x**3 + x**2) * (1-y) ] ) )
mesh.coordinates.assign(f_mesh)

xs = [mesh.coordinates.dat.data[i][0] for i in range(0,66)]
ys = [mesh.coordinates.dat.data[i][1] for i in range(0,66)]

plt.scatter(xs, ys)
plt.show()

########################### define function spaces #####################################
CG_1 = FiniteElement("CG", interval, 1)
DG_0 = FiniteElement("DG", interval, 0)
P1P0 = TensorProductElement(CG_1, DG_0)
RT_horiz = HDivElement(P1P0)
P0P1 = TensorProductElement(DG_0, CG_1)
RT_vert = HDivElement(P0P1)
element = RT_horiz + RT_vert

Sigma = FunctionSpace(mesh, element)
VD = FunctionSpace(mesh, "DG", 0)

W = Sigma * VD

#############define problem ########################################################
sigma, u = TrialFunctions(W)
tau, v = TestFunctions(W)

#boundary conditions_ Neumann at top and bottom for sigma=W.sub(0)
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



######################################## hybridized version of Helmholtz' problem with Hybridization PC ######################################

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

# will be the solution
wh = Function(W)

problem = LinearVariationalProblem(a, L, wh, bcs=[bc0, bc1, bc2, bc3])

solver = LinearVariationalSolver(problem, solver_parameters=hybrid_parameters)

solver.solve()

sigmah, uh = wh.split()

file2 = File("testPiolaHybridPC.pvd")
file2.write(sigmah, uh)

fig, axes = plt.subplots()

quiver(sigmah, axes=axes)
axes.set_aspect("equal")
axes.set_title("$\sigma$")
plt.show()

######################################## hybridized version of Helmholtz' problem with SCPC ######################################


Sigmahat = FunctionSpace(mesh, BrokenElement(Sigma.ufl_element()))
V = FunctionSpace(mesh, VD.ufl_element())
T = FunctionSpace(mesh, FiniteElement("HDiv Trace", mesh.ufl_cell(), degree=0))
W_hybrid = Sigmahat * V * T

n = FacetNormal(mesh)
sigmahat, uhat, lambdar = TrialFunctions(W_hybrid)
tauhat, vhat, gammar = TestFunctions(W_hybrid)

wh = Function(W_hybrid)

a_hybrid = (inner(sigmahat, tauhat) * dx + div(tauhat) * uhat * dx + div(sigmahat) * vhat * dx + vhat * uhat * dx
            + inner(tauhat, n) * lambdar * (ds_b + ds_t + ds_v)
            + inner(sigmahat, n) * gammar * (ds_b + ds_t + ds_v)
            + jump(tauhat, n=n) * lambdar('+') * (dS_h + dS_v)
            + jump(sigmahat, n=n) * gammar('+') * (dS_h + dS_v))

L = -f * vhat * dx

scpc_parameters = {"ksp_type": "preonly", "pc_type": "lu"}

solve(a_hybrid == L, wh, solver_parameters={"ksp_type": "gmres", "mat_type": "matfree",
                                            "pc_type": "python", "pc_python_type": "firedrake.SCPC",
                                            "condensed_field": scpc_parameters,
                                            "pc_sc_eliminate_fields": "0,1"})

sigmah, uh, lamdah = wh.split()

file2 = File("testPiolaSCPC.pvd")
file2.write(sigmah, uh)

fig, axes = plt.subplots()

quiver(sigmah, axes=axes)
axes.set_aspect("equal")
axes.set_title("$\sigma$")
plt.show()