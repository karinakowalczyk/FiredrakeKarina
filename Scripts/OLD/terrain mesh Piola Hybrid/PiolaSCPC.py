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

######################################## hybridized version of Helmholtz' problem with SCPC ###########################


Sigmahat = FunctionSpace(mesh, BrokenElement(Sigma.ufl_element()))
V = FunctionSpace(mesh, VD.ufl_element())
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

scpc_parameters = {"ksp_type": "preonly", "pc_type": "lu"}

solve(a_hybrid == L, wh, solver_parameters={"ksp_type": "gmres", "mat_type": "matfree",
                                            "pc_type": "python", "pc_python_type": "firedrake.SCPC",
                                            "condensed_field": scpc_parameters,
                                            "pc_sc_eliminate_fields": "0,1"})

sigmah, uh, lamdah = wh.split()

file2 = File("PiolaSCPC.pvd")
file2.write(sigmah, uh)

fig, axes = plt.subplots()

quiver(sigmah, axes=axes)
axes.set_aspect("equal")
axes.set_title("$\sigma$")
plt.show()