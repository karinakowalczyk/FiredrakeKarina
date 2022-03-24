from petsc4py import PETSc
#PETSc.Sys.popErrorHandler()
import firedrake as fd

dT = fd.Constant(0)

nlayers = 40 # horizontal layers
columns = 40  # number of columns
L = 3.0e5
m = fd.PeriodicIntervalMesh(columns, L)

cs = fd.Constant(100.)
f = fd.Constant(1.0)
N = fd.Constant(1.0e-2)
U = fd.Constant(20.0)

# build volume mesh
H = 1.0e4  # Height position of the model top
mesh = fd.ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

Vc = mesh.coordinates.function_space()
x, y = fd.SpatialCoordinate(mesh)
#f_mesh = Function(Vc).interpolate(as_vector([x, y + ( 0.25 * x**4 -x**3 + x**2) * (1-y) ] ) )
#f_mesh = fd.Function(Vc).interpolate(fd.as_vector([x, 2*y] ))
f_mesh = fd.Function(Vc).interpolate(fd.as_vector([x,y - (1/H)*fd.exp(-x**2/2)*((y-0.5*H)**2 -0.25* H**2 )] ) )
mesh.coordinates.assign(f_mesh)

H = fd.Constant(H)

family = "CG"
horizontal_degree = 0
vertical_degree = 0
S1 = fd.FiniteElement(family, fd.interval, horizontal_degree+1)
S2 = fd.FiniteElement("DG", fd.interval, horizontal_degree)

# vertical base spaces
T0 = fd.FiniteElement("CG", fd.interval, vertical_degree+1)
T1 = fd.FiniteElement("DG", fd.interval, vertical_degree)
Tlinear = fd.FiniteElement("CG", fd.interval, 1)

# build spaces V2, V3, Vt
V2h_elt = fd.HDiv(fd.TensorProductElement(S1, T1))
V2t_elt = fd.TensorProductElement(S2, T0)
V3_elt = fd.TensorProductElement(S2, T1)
V2v_elt = fd.HDiv(V2t_elt)
V2v_elt_Broken = fd.BrokenElement(fd.HDiv(V2t_elt))
#V2_elt = V2h_elt + V2v_elt
V2_elt = fd.EnrichedElement(V2h_elt, V2v_elt_Broken)
VT_elt = fd.TensorProductElement(S2, Tlinear)

V1 = fd.FunctionSpace(mesh, V2_elt, name="HDiv")
remapped = fd.WithMapping(V2_elt, "identity")
V1 = fd.FunctionSpace(mesh, remapped, name="HDiv")

V2 = fd.FunctionSpace(mesh, V3_elt, name="DG")
Vt = fd.FunctionSpace(mesh, V2t_elt, name="Temperature")
Vv = fd.FunctionSpace(mesh, V2v_elt, name="Vv")

T = fd.FunctionSpace(mesh, VT_elt)

W = V1 * V2 * Vt * T #velocity, pressure, temperature, trace of velocity

Un = fd.Function(W)
Unp1 = fd.Function(W)

x, z = fd.SpatialCoordinate(mesh)

xc = L/2
a = fd.Constant(5000)

un, Pin, bn, lamdar = Un.split()
bn.interpolate(fd.sin(fd.pi*z/H)/(1+(x-xc)**2/a**2))

#The timestepping solver
un, Pin, bn, lamdan = fd.split(Un)
unp1, Pinp1, bnp1, lamdanp1 = fd.split(Unp1)
unph = 0.5*(un + unp1)
bnph = 0.5*(bn + bnp1)
lamdanph = 0.5*(lamdan + lamdanp1)
Pinph = 0.5*(Pin + Pinp1)
#Ubar = fd.as_vector([U, 0])   #linear case
Ubar = unph
n = fd.FacetNormal(mesh)
unn = 0.5*(fd.dot(Ubar, n) + abs(fd.dot(Ubar, n)))

k = fd.as_vector([0, 1])
def theta_eqn(q):
    return (
        q*(bnp1 - bn)*fd.dx -
        dT*fd.div(q*Ubar)*bnph*fd.dx +
        dT*N**2*q*fd.inner(k, unph)*fd.dx +
        dT*fd.jump(q)*(unn('+')*bnph('+')
        - unn('-')*bnph('-'))*(fd.dS_v + fd.dS_h)
    )

def pi_eqn(q):
    return (
        q*(Pinp1 - Pin)*fd.dx -
        dT*fd.inner(fd.grad(q), Ubar*Pinph)*fd.dx
        + dT*fd.jump(q)*(unn('+')*Pinph('+') - unn('-')*Pinph('-'))*(fd.dS_v + fd.dS_h)
        + cs**2*dT*q*fd.div(unph)*fd.dx
    )

def u_eqn(w, gammar):
    return (
        fd.inner(w, unp1 - un)*fd.dx


        - dT*fd.inner(fd.grad(w), fd.outer(unph, Ubar))*fd.dx

        +dT*fd.dot(fd.jump(w), unn('+')*unph('+')
                         - unn('-')*unph('-'))*(fd.dS_v + fd.dS_h)

        + dT*(fd.jump(unp1, n=n) * gammar('+') * (fd.dS_h) + fd.jump(w, n=n) * lamdanp1('+') * (fd.dS_h))
        + dT*(fd.inner(unp1, n) * gammar * fd.ds_tb + fd.inner(w, n) * lamdanp1 * (fd.ds_tb))

        -dT*fd.div(w)*Pinph*fd.dx - dT*fd.inner(w, k)*bnph*fd.dx
        )

w, phi, q, gammar = fd.TestFunctions(W)
gamma = fd.Constant(10000.0)
eqn = u_eqn(w, gammar) + theta_eqn(q) + pi_eqn(phi) + gamma*pi_eqn(fd.div(w))

nprob = fd.NonlinearVariationalProblem(eqn, Unp1)
luparams = {'snes_monitor':None,
    'mat_type':'aij',
    'ksp_type':'preonly',
    'pc_type':'lu',
    'pc_factor_mat_solver_type':'mumps'}

sparameters = {
    "mat_type":"matfree",
    'snes_monitor': None,
    "ksp_type": "fgmres",
    #"ksp_view": None,
    "ksp_gmres_modifiedgramschmidt": None,
    'ksp_converged_reason': None,
    'ksp_monitor': None,
    "ksp_rtol": 1e-8,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_0_fields": "0,2,3",
    "pc_fieldsplit_1_fields": "1",
    "pc_fieldsplit_schur_fact_type": "full",
    "pc_fieldsplit_off_diag_use_amat": True,
}

bottomright = {
    "ksp_type": "gmres",
    "ksp_max_it": 3,
    "pc_type": "python",
    "pc_python_type": "firedrake.MassInvPC",
    "Mp_pc_type": "bjacobi",
    "Mp_sub_pc_type": "ilu"
}

sparameters["fieldsplit_1"] = bottomright

topleft_LU = {
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled_pc_type": "lu",
    "assembled_pc_factor_mat_solver_type": "mumps"
}

topleft_LS = {
    'ksp_type': 'preonly',
    'pc_type': 'python',
    "pc_python_type": "firedrake.AssembledPC",
    'assembled_pc_type': 'python',
    'assembled_pc_python_type': 'firedrake.ASMStarPC',
    "assembled_pc_star_sub_pc_type": "lu",
    'assembled_pc_star_dims': '0',
    'assembled_pc_star_sub_pc_factor_mat_solver_type' : 'mumps'
    #'assembled_pc_linesmooth_star': '1'
}

sparameters["fieldsplit_0"] = topleft_LS

nsolver = fd.NonlinearVariationalSolver(nprob,
                                        solver_parameters=sparameters)

name = "Results/New/gw_imp"
file_gw = fd.File(name+'.pvd')
un, Pin, bn, lamdan = Un.split()
file_gw.write(un, Pin, bn)
Unp1.assign(Un)

dt = 600.
dumpt = 600.
tdump = 0.
dT.assign(dt)
tmax = 3600.
print('tmax', tmax, 'dt', dt)
t = 0.
while t < tmax - 0.5*dt:
    print(t)
    t += dt
    tdump += dt

    nsolver.solve()
    Un.assign(Unp1)

    if tdump > dumpt - dt*0.5:
        file_gw.write(un, Pin, bn)
        tdump -= dumpt
