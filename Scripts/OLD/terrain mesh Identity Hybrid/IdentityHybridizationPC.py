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
W = Sigma * V
########################## set up problem ####################################################



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

####################### solve ##############################################################
#will be the solution
wh = Function(W)

problem = LinearVariationalProblem(a, L, wh, bcs=[bc0,bc1, bc2, bc3])

solver = LinearVariationalSolver(problem, solver_parameters=hybrid_parameters)

solver.solve()
sigmah, uh = wh.split()

#output/plot solution
file0 = File("IdentityHybridizationPC.pvd")
#can't write sigmah to file???
file0.write(sigmah,uh)
fig, axes = plt.subplots()
quiver(sigmah, axes=axes)
axes.set_aspect("equal")
axes.set_title("$\sigma$")
