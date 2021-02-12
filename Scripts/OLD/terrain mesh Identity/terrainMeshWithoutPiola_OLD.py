import matplotlib.pyplot as plt
from firedrake import *


######################### define mesh ##############################

m = IntervalMesh(10,2)
mesh = ExtrudedMesh(m, 5, extrusion_type='uniform')

Vc = mesh.coordinates.function_space()
x, y = SpatialCoordinate(mesh)
f = Function(Vc).interpolate(as_vector([x, y + ( 0.25 * x**4 -x**3 + x**2) * (1-y) ] ) )
mesh.coordinates.assign(f)

xs = [mesh.coordinates.dat.data[i][0] for i in range(0,66)]
ys = [mesh.coordinates.dat.data[i][1] for i in range(0,66)]

plt.scatter(xs, ys)
plt.show()

################# define function spaces ############################

CG_1 = FiniteElement("CG", interval, 1)
DG_0 = FiniteElement("DG", interval, 0)
P1P0 = TensorProductElement(CG_1, DG_0)
#RT_horiz = HDivElement(P1P0)
RT_horiz = P1P0

P0P1 = TensorProductElement(DG_0, CG_1)
#RT_vert = HDivElement(P0P1)
RT_vert = P0P1

Sigma_horiz = FunctionSpace(mesh, RT_horiz)
Sigma_vert = FunctionSpace(mesh, RT_vert)
V = FunctionSpace(mesh, "DG", 0)

#################### set up problem ################################

Sigmahat_horiz = FunctionSpace(mesh, BrokenElement(Sigma_horiz.ufl_element()))  # do I need broken element here??
Sigmahat_vert = FunctionSpace(mesh, BrokenElement(Sigma_vert.ufl_element()))
V = FunctionSpace(mesh, V.ufl_element())
T = FunctionSpace(mesh, FiniteElement("HDiv Trace", mesh.ufl_cell(), degree=0))
W_hybrid = Sigmahat_horiz * Sigmahat_vert * V * T

n = FacetNormal(mesh)

sigmahat_horiz, sigmahat_vert, uhat, lambdar = TrialFunctions(W_hybrid)
tauhat_horiz, tauhat_vert, vhat, gammar = TestFunctions(W_hybrid)

sigmahat = as_vector([0, 1]) * sigmahat_vert + as_vector([1, 0]) * sigmahat_horiz
tauhat = as_vector([0, 1]) * tauhat_vert + as_vector([1, 0]) * tauhat_horiz

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

################################### solve problem ##########################################

parameters = {"ksp_type": "gmres", "ksp_monitor": None,
              "pc_type": "lu", "mat_type": "aij",
              "pc_factor_mat_solver_type": "mumps"}

solve(a_hybrid == L, wh, bcs=[], solver_parameters=parameters)

sigmah_horiz, sigmah_vert, uh, lambdah = wh.split()
sigmah = as_vector([0,1])*sigmah_vert + as_vector([1,0])*sigmah_horiz

V_plot = VectorFunctionSpace(mesh, "DG", 0)
sigmah_plot = Function(V_plot).interpolate(sigmah)

#output/plot solution
file0 = File("test.pvd")
#can't write sigmah to file???
#file0.write(sigmah_plot,uh)

fig, axes = plt.subplots()
quiver(sigmah, axes=axes)
axes.set_aspect("equal")
axes.set_title("$\sigma$")

plt.show()

############################################################################

############################################################################