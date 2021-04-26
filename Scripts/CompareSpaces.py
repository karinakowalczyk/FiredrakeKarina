import matplotlib.pyplot as plt
from firedrake import *
import petsc4py.PETSc as PETSc
PETSc.Sys.popErrorHandler()


##############   define mesh ######################################################################

m = IntervalMesh(20,2)
mesh = ExtrudedMesh(m, 10, extrusion_type='uniform')

Vc = mesh.coordinates.function_space()
x, y = SpatialCoordinate(mesh)
f_mesh = Function(Vc).interpolate(as_vector([x, y + ( 0.25 * x**4 -x**3 + x**2) * (1-y) ] ) )
mesh.coordinates.assign(f_mesh)

xs = [mesh.coordinates.dat.data[i][0] for i in range(0,66)]
ys = [mesh.coordinates.dat.data[i][1] for i in range(0,66)]

plt.scatter(xs, ys)

################################################################################################################
##################### NEW Space ###############################################################################
################################################################################################################


def L2ProjectionNewSpace(mesh):
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
    V = FunctionSpace(mesh, remapped)

    T = FunctionSpace(mesh, P0P1)

    W_hybrid = V * T

    ######################################################################################################################


    uexact = as_vector((1 + x, 0))

    n = FacetNormal(mesh)

    u, lambdar = TrialFunctions(W_hybrid)
    v, gammar = TestFunctions(W_hybrid)
    wh = Function(W_hybrid)

    aL2 = inner(uexact - u,v)*dx + jump(v, n=n) * lambdar('+') * (dS_h) + jump(u, n=n) * gammar('+')*(dS_h)

    bc0 = DirichletBC(W_hybrid.sub(1), 0, 1)
    bc1 = DirichletBC(W_hybrid.sub(1), 0, 2)
    bc2 = DirichletBC(W_hybrid.sub(1), 0, 'top')
    bc3 = DirichletBC(W_hybrid.sub(1), 0, 'bottom')

    parameters = {"pc_type":"lu", "pc_factor_mat_solver_type":"mumps", "ksp_type":"preonly"}

    solve(lhs(aL2) == rhs(aL2), wh, bcs=[bc0, bc1, bc2, bc3], solver_parameters=parameters)

    up, lamdah = wh.split()

    fig, axes = plt.subplots()
    quiver(up, axes=axes)

    normdifference = norm(uexact - up, norm_type="L2")
    print("norm differnce L2Proj to new Space =" + str(normdifference))

    fig.savefig("../Results/ResultsCompareSpaces/L2projNew.png")

    return up

################################################################################################################
##################### Piola mapped Space ###############################################################################
################################################################################################################

def L2ProjectionPiola (mesh):

    ############## define function spaces ##########################################################################

    CG_1 = FiniteElement("CG", interval, 1)
    DG_0 = FiniteElement("DG", interval, 0)
    P1P0 = TensorProductElement(CG_1, DG_0)
    RT_horiz = HDivElement(P1P0)
    RT_horiz_broken = BrokenElement(RT_horiz)
    P0P1 = TensorProductElement(DG_0, CG_1)
    RT_vert = HDivElement(P0P1)
    RT_vert_broken = BrokenElement(RT_vert)
    full = EnrichedElement(RT_horiz, RT_vert)

    V_Piola = FunctionSpace(mesh, full)


    ######################################################################################################################

    uexact = as_vector((1 + x, 0))

    n = FacetNormal(mesh)

    u = TrialFunction(V_Piola)
    v = TestFunction(V_Piola)
    up = Function(V_Piola)

    aL2 = inner(uexact - u,v)*dx


    parameters = {"pc_type":"lu", "pc_factor_mat_solver_type":"mumps", "ksp_type":"preonly"}

    solve(lhs(aL2) == rhs(aL2), up, bcs=[], solver_parameters=parameters)


    ### output

    fig, axes = plt.subplots()
    quiver(up, axes=axes)

    normdifference = norm(uexact - up, norm_type="L2")
    print("norm differnce L2Proj to Piola mapped space =" + str(normdifference))

    fig.savefig("../Results/ResultsCompareSpaces/L2ProjPiola.png")

    return up

####################################################################################################################
#Consider exact solution u = x of Helmholtz problem
####################################################################################################################


def SolveWithPiola(mesh):
    CG_1 = FiniteElement("CG", interval, 2)
    DG_0 = FiniteElement("DG", interval, 1)
    P1P0 = TensorProductElement(CG_1, DG_0)
    RT_horiz = HDivElement(P1P0)
    P0P1 = TensorProductElement(DG_0, CG_1)
    RT_vert = HDivElement(P0P1)
    element = RT_horiz + RT_vert

    # Sigma = FunctionSpace(mesh, "RTCF", 1)
    Sigma = FunctionSpace(mesh, element)
    VD = FunctionSpace(mesh, "DQ", 1)


    W = Sigma * VD

    n = FacetNormal(mesh)
    sigma, u = TrialFunctions(W)
    tau, v= TestFunctions(W)

    wh = Function(W)


    sigmaexact = as_vector((1 + x, 0))
    uexact = 0.5 * x**2 + x
    f = -1

    V_plot = VectorFunctionSpace(mesh, "DG", 1)
    sigmaexact_plot = Function(V_plot).project(sigmaexact)
    uexact_plot = Function(VD).project(uexact)
    normsigmaexactY = norm(sigmaexact_plot[1], norm_type="L2")
    print("norm Y Component sigmaexact =" + str(normsigmaexactY))

    fig, axes = plt.subplots()
    quiver(sigmaexact_plot, axes=axes)
    axes.set_aspect("equal")
    axes.set_title("$\sigma$ exact")
    fig.savefig("../Results/ResultsCompareSpaces/sigmaExact.png")

    bc0 = DirichletBC(W.sub(0), sigmaexact, "bottom")
    bc1 = DirichletBC(W.sub(0), sigmaexact, "top")
    bc2 = DirichletBC(W.sub(0), sigmaexact, 1)
    bc3 = DirichletBC(W.sub(0), sigmaexact, 2)

    a_hybrid_Piola = (inner(sigma, tau) * dx + div(tau) * u * dx
                      - div(sigma) * v * dx
                      - f * v* dx(degree=(5, 5)))

    parameters = {"ksp_type": "gmres", "ksp_monitor": None, "pc_type": "lu", "mat_type": "aij",
                  "pc_factor_mat_solver_type": "mumps"}

    solve(lhs(a_hybrid_Piola) == rhs(a_hybrid_Piola), wh, bcs=[bc0, bc1, bc2, bc3], solver_parameters=parameters)


    sigmah, uh = wh.split()

    fig, axes = plt.subplots()
    quiver(sigmah, axes=axes)
    axes.set_aspect("equal")
    axes.set_title("$\sigma$ with Piola")
    fig.savefig("../Results/ResultsCompareSpaces/sigmaPiola.png")

    file = File("../Results/ResultsCompareSpaces/Piola.pvd")
    file.write(sigmah, uh)

    normSigmahY = norm(sigmah[1], norm_type="L2")

    print("norm Y Component with Piola =" + str(normSigmahY))

    return (uh, sigmah, normSigmahY)

def SolveWithIdentity_BrokenVertical(mesh):

    ########################### define function spaces #####################################

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
    # T = FunctionSpace(mesh, FiniteElement("HDiv Trace", mesh.ufl_cell(), degree=0))

    W = Sigmahat * V * T

    ############# define problem ########################################################

    n = FacetNormal(mesh)

    sigmahat, uhat, lambdar = TrialFunctions(W)
    tauhat, vhat, gammar = TestFunctions(W)

    wh = Function(W)

    sigmaexact = as_vector((1 + x, 0))
    uexact = 0.5 * x ** 2 + x
    f = -1

    bc0 = DirichletBC(W.sub(0), sigmaexact, 1)
    bc1 = DirichletBC(W.sub(0), sigmaexact, 2)

    a_hybrid_Identity_BrokenVert = (inner(sigmahat, tauhat) * dx + div(tauhat) * uhat * dx
                - div(sigmahat) * vhat * dx
                + inner(tauhat, n) * lambdar * (ds_b + ds_t)
                + inner(sigmahat, n) * gammar * (ds_b + ds_t)
                - inner(sigmaexact, n) * gammar * (ds_b(degree=(5, 5)) + ds_t(degree=(5, 5)) )
                + jump(tauhat, n=n) * lambdar('+') * (dS_h)
                + jump(sigmahat, n=n) * gammar('+') * (dS_h)
                - f * vhat * dx(degree=(5, 5)))


    wh = Function(W)

    #######################################  hybridized with SCPC   #####################################################

    scpc_parameters = {"ksp_type": "preonly", "pc_type": "lu"}
    parameters = {"pc_type":"lu", "pc_factor_mat_solver_type":"mumps", "ksp_type":"preonly"}

    solve(lhs(a_hybrid_Identity_BrokenVert) == rhs(a_hybrid_Identity_BrokenVert), wh, solver_parameters=parameters, bcs = [bc0, bc1])

    sigmah, uh, lamdah = wh.split()

    fig, axes = plt.subplots()
    quiver(sigmah, axes=axes)
    axes.set_aspect("equal")
    axes.set_title("$\sigma$ broken in vertical")
    fig.savefig("../Results/ResultsCompareSpaces/sigmaNewSpace.png")

    file2 = File("../Results/ResultsCompareSpaces/Identity_BrokenVert.pvd")
    file2.write(sigmah, uh)

    normSigmahY = norm(sigmah[1], norm_type="L2")
    print("norm Y Component with new space =" + str(normSigmahY))

    return (uh, sigmah, normSigmahY)



SolveWithPiola(mesh)
SolveWithIdentity_BrokenVertical(mesh)
L2ProjectionNewSpace(mesh)
L2ProjectionPiola(mesh)