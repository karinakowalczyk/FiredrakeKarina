from petsc4py import PETSc
PETSc.Sys.popErrorHandler()
import firedrake as fd

dT = fd.Constant(0)

nlayers = 50  # horizontal layers
columns = 110  # number of columns
L = 3.0e5
m = fd.PeriodicIntervalMesh(columns, L)

cs = fd.Constant(300.)
f = fd.Constant(1.0)
N = fd.Constant(1.0e-2)
U = fd.Constant(20.)

# build volume mesh
H = 1.0e4  # Height position of the model top
mesh = fd.ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

Vc = mesh.coordinates.function_space()
x, y = fd.SpatialCoordinate(mesh)
profile = ((1000)**2)/((x-L/2)**2 + (1000)**2)
f_mesh = fd.Function(Vc).interpolate(fd.as_vector([x, y + (profile * (1-y/H)) ] ) )
#f_mesh = fd.Function(Vc).interpolate(fd.as_vector([x, 2*y] ))
#f_mesh = fd.Function(Vc).interpolate(fd.as_vector([x,y - (1/H)*fd.exp(-x**2/2)*((y-0.5*H)**2 -0.25* H**2 )] ) )
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
#bn.interpolate(fd.sin(fd.pi*z/H)/(1+(x-xc)**2/a**2))
#bn.interpolate(fd.Constant(0.0001))
un.project(fd.as_vector([U,0]))

#The timestepping solver
un, Pin, bn, lamdan = fd.split(Un)
unp1, Pinp1, bnp1, lamdanp1 = fd.split(Unp1)

unph = 0.5*(un + unp1)
bnph = 0.5*(bn + bnp1)
lamdanph = 0.5*(lamdan + lamdanp1)
Pinph = 0.5*(Pin + Pinp1)
#Ubar = fd.as_vector([U, 0])-
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
        + dT*(fd.inner(Ubar +unp1, n) * gammar * fd.ds_tb + fd.inner(w, n) * lamdanp1 * (fd.ds_tb))

        -dT*fd.div(w)*Pinph*fd.dx - dT*fd.inner(w, k)*bnph*fd.dx
        )


w, phi, q, gammar = fd.TestFunctions(W)
gamma = fd.Constant(1000.0)
eqn = u_eqn(w, gammar) + theta_eqn(q) + pi_eqn(phi) + gamma*pi_eqn(fd.div(w))
assemble(eqn)

nprob = fd.NonlinearVariationalProblem(eqn, Unp1)

v_basis = fd.VectorSpaceBasis(constant=True)
nullspace = fd.MixedVectorSpaceBasis(W, [v_basis, W.sub(1)])

luparams = {'snes_monitor':None,
    'mat_type':'aij',
    'ksp_type':'preonly',
    'pc_type':'lu',
    'pc_factor_mat_solver_type':'mumps'}

sparameters = {
    "mat_type":"matfree",
    'snes_monitor': None,
    "ksp_type": "fgmres",
    "ksp_gmres_modifiedgramschmidt": None,
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

topleft_MG = {
    "ksp_type": "preonly",
    "ksp_max_it": 3,
    "pc_type": "mg",
    "mg_coarse_ksp_type": "preonly",
    "mg_coarse_pc_type": "python",
    "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "mg_coarse_assembled_pc_type": "lu",
    "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
    "mg_levels_ksp_type": "gmres",
    "mg_levels_ksp_max_it": 3,
    "mg_levels_pc_type": "python",
    "mg_levels_pc_python_type": "firedrake.PatchPC",
    "mg_levels_patch_pc_patch_save_operators": True,
    "mg_levels_patch_pc_patch_partition_of_unity": False,
    "mg_levels_patch_pc_patch_sub_mat_type": "seqaij",
    "mg_levels_patch_pc_patch_construct_type": "star",
    "mg_levels_patch_pc_patch_multiplicative": False,
    "mg_levels_patch_pc_patch_symmetrise_sweep": False,
    "mg_levels_patch_pc_patch_construct_dim": 0,
    "mg_levels_patch_sub_ksp_type": "preonly",
    "mg_levels_patch_sub_pc_type": "lu",
}
sparameters["fieldsplit_0"] = topleft_LS

nsolver = fd.NonlinearVariationalSolver(nprob,
                                        solver_parameters=sparameters)

name = "Results/MountainFlow/gw_imp"
file_gw = fd.File(name+'.pvd')
un, Pin, bn, lamdan = Un.split()
file_gw.write(un, Pin, bn)


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
