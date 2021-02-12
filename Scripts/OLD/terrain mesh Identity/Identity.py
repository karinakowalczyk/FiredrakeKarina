import matplotlib.pyplot as plt
from firedrake import *
import petsc4py.PETSc as PETSc
PETSc.Sys.popErrorHandler()


################### mesh #####################################################################

m = IntervalMesh(10,2)
mesh = ExtrudedMesh(m, 5, extrusion_type='uniform')

Vc = mesh.coordinates.function_space()
x, y = SpatialCoordinate(mesh)
f = Function(Vc).interpolate(as_vector([x, y + ( 0.25 * x**4 -x**3 + x**2) * (1-y) ] ) )
mesh.coordinates.assign(f)

xs = [mesh.coordinates.dat.data[i][0] for i in range(0,66)]
ys = [mesh.coordinates.dat.data[i][1] for i in range(0,66)]

plt.scatter(xs, ys)

############## function spaces ################################################################

element = FiniteElement("RTCF", cell="quadrilateral", degree=1)
element._mapping = 'identity'
Sigma = FunctionSpace(mesh, element)
V = FunctionSpace(mesh, "DG", 0)

########################## set up problem ####################################################

Sigmahat = FunctionSpace(mesh, BrokenElement(Sigma.ufl_element()))  # do I need broken element here??
V = FunctionSpace(mesh, V.ufl_element())
T = FunctionSpace(mesh, FiniteElement("HDiv Trace", mesh.ufl_cell(), degree=0))
W_hybrid = Sigmahat * V * T

n = FacetNormal(mesh)

sigmahat, uhat, lambdar = TrialFunctions(W_hybrid)
tauhat, vhat, gammar = TestFunctions(W_hybrid)

wh = Function(W_hybrid)

f = 10 * exp(-100 * ((x - 1) ** 2 + (y - 0.5) ** 2))

a_hybrid = (inner(sigmahat, tauhat) * dx + div(tauhat) * uhat * dx
            + div(sigmahat) * vhat * dx + vhat * uhat * dx
            + inner(tauhat, n) * lambdar * (ds_b + ds_t + ds_v)
            + inner(sigmahat, n) * gammar * (ds_b + ds_t + ds_v)
            + jump(tauhat, n=n) * lambdar('+') * (dS_h + dS_v)
            + jump(sigmahat, n=n) * gammar('+') * (dS_h + dS_v))

L = -f * vhat * dx

######################### solve ###############################################################

#will be the solution
wh = Function(W_hybrid)

#solve

parameters = {"ksp_type":"gmres", "ksp_monitor":None,
              "pc_type":"lu", "mat_type":"aij",
              "pc_factor_mat_solver_type":"mumps" }

solve(a_hybrid == L, wh, bcs = [], solver_parameters=parameters)

sigmah, uh, lambdah = wh.split()

#output/plot solution
file0 = File("Identity.pvd")
#can't write sigmah to file???
file0.write(sigmah,uh)
fig, axes = plt.subplots()
quiver(sigmah, axes=axes)
axes.set_aspect("equal")
axes.set_title("$\sigma$")
