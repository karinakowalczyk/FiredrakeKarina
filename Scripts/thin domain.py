import matplotlib.pyplot as plt
from firedrake import *
#import petsc4py.PETSc as PETSc
#PETSc.Sys.popErrorHandler()


##############   define mesh ######################################################################

m = IntervalMesh(10,1)
mesh = ExtrudedMesh(m, 5, layer_height=0.01, extrusion_type='uniform')

Vc = mesh.coordinates.function_space()
x, y = SpatialCoordinate(mesh)
f_mesh = Function(Vc).interpolate(as_vector([x, y + ( 0.25 * x**4 -x**3 + x**2) * (1-y) ] ) )
mesh.coordinates.assign(f_mesh)

xs = [mesh.coordinates.dat.data[i][0] for i in range(0,66)]
ys = [mesh.coordinates.dat.data[i][1] for i in range(0,66)]

plt.scatter(xs, ys)

############## define function spaces ##########################################################################

CG_1 = FiniteElement("CG", interval, 1)
DG_0 = FiniteElement("DG", interval, 0)
P1P0 = TensorProductElement(CG_1, DG_0)
RT_horiz = HDivElement(P1P0)
RT_horiz_broken = BrokenElement(RT_horiz)
P0P1 = TensorProductElement(DG_0, CG_1)
RT_vert = HDivElement(P0P1)
RT_vert_broken = BrokenElement(RT_vert)
full = EnrichedElement(RT_horiz, RT_vert_broken)
Sigma = FunctionSpace(mesh, full)
remapped = WithMapping(full, "identity")
Sigmahat = FunctionSpace(mesh, remapped)

V = FunctionSpace(mesh, "DQ", 0)
T = FunctionSpace(mesh, P0P1)

W_hybrid = Sigmahat * V * T

######################################################################################################################

#f = 10 * exp(-100 * ((x - 1) ** 2 + (y - 0.5) ** 2))

uexact = cos(2* pi * x) * cos(2 * pi * y)
sigmaexact = -2* pi * as_vector((sin(2*pi*x)*cos(2*pi*y), cos(2*pi*x)*sin(2*pi*y)))
f = (1 + 8*pi*pi) * uexact

def eqn_sigmahat (tauhat):
    return( inner(sigmahat, tauhat) * dx + div(tauhat) * uhat * dx
               + inner(tauhat, n) * lambdar * (ds_b + ds_t )
               + jump(tauhat, n=n) * lambdar('+') * (dS_h)
                )

def eqn_uhat(vhat):
    return( - div(sigmahat) * vhat * dx + vhat * uhat * dx
                )

def eqn_lamda(gammar):
    return( + inner(sigmahat - sigmaexact, n) * gammar * (ds_b + ds_t)
                + jump(sigmahat, n=n) * gammar('+') * (dS_h)
                )

n = FacetNormal(mesh)

sigmahat, uhat, lambdar = TrialFunctions(W_hybrid)
tauhat, vhat, gammar = TestFunctions(W_hybrid)
wh = Function(W_hybrid)


sparameters = {
    "mat_type":"matfree",
    'snes_monitor': None,
    "ksp_type": "fgmres",
    "ksp_gmres_modifiedgramschmidt": None,
    'ksp_monitor': None,
    "ksp_rtol": 1e-8,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_0_fields": "0,2",
    "pc_fieldsplit_1_fields": "1",
    "pc_fieldsplit_schur_fact_type": "full",
    "pc_fieldsplit_off_diag_use_amat": True,
}

bottomright = {
    "ksp_type": "gmres",
    "ksp_max_it": 3,
    "ksp_monitor": None,
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


topleft_LS = {'pc_type': 'python',
              'pc_python_type': 'firedrake.ASMLinesmoothPC',
              'pc_linesmooth_codims': '1'
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
sparameters["fieldsplit_0"] = topleft_LU


for count in range(10):

    constant = 1000.0 + count*1000.0

    gamma = Constant(constant)

    a = eqn_sigmahat(tauhat) + eqn_uhat(vhat) + eqn_lamda(gammar) + gamma * eqn_uhat(div(tauhat))

    a-=  f * vhat * dx(degree=(5, 5))

    bc0 = DirichletBC(W_hybrid.sub(0), sigmaexact, 1)
    bc1 = DirichletBC(W_hybrid.sub(0), sigmaexact, 2)


    ctx = {"mu": 1/gamma}

    prob = LinearVariationalProblem(lhs(a), rhs(a), wh, bcs = [bc0, bc1])
    solver = LinearVariationalSolver(prob, solver_parameters=sparameters, appctx = ctx)

    solver.solve()

    sigmah, uh, lamdah = wh.split()

    fig, axes = plt.subplots()
    quiver(sigmah, axes=axes)
    axes.set_aspect("equal")
    axes.set_title("$\sigma$")

    fig.savefig("../Results/sigmaAugLagrangian_"+ str(constant) + ".png")

    file = File("../Results/augmentedLagrangian.pvd")
    file.write(sigmah, uh)