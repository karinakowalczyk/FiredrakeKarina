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
VD = FunctionSpace(mesh, "DG", 0)

W = Sigma * VD

#############define problem ########################################################
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
file0 = File("Piola.pvd")
file0.write(sigmah, uh)

fig, axes = plt.subplots()
quiver(sigmah, axes=axes)
axes.set_aspect("equal")
axes.set_title("$\sigma$")
plt.show()
