import matplotlib.pyplot as plt
from firedrake import *
#import petsc4py.PETSc as PETSc
#PETSc.Sys.popErrorHandler()


def SolveWithPiola(mesh):

    CG_1 = FiniteElement("CG", interval, 1)
    DG_0 = FiniteElement("DG", interval, 0)
    P1P0 = TensorProductElement(CG_1, DG_0)
    RT_horiz = HDivElement(P1P0)
    P0P1 = TensorProductElement(DG_0, CG_1)
    RT_vert = HDivElement(P0P1)
    element = RT_horiz + RT_vert

    # Sigma = FunctionSpace(mesh, "RTCF", 1)
    Sigma = FunctionSpace(mesh, element)
    VD = FunctionSpace(mesh, "DQ", 0)

    Sigmahat = FunctionSpace(mesh, BrokenElement(Sigma.ufl_element()))
    V = FunctionSpace(mesh, VD.ufl_element())
    T = FunctionSpace(mesh, FiniteElement("HDiv Trace", mesh.ufl_cell(), degree=0))
    W_hybrid = Sigmahat * V * T

    n = FacetNormal(mesh)
    sigmahat, uhat, lambdar = TrialFunctions(W_hybrid)
    tauhat, vhat, gammar = TestFunctions(W_hybrid)

    wh = Function(W_hybrid)

    f = (1 + 4 * pi**2) * cos(2*pi*x)
    sigmaexact = as_vector((-2*pi*sin(2*pi*x), 0))
    uexact = cos(2*pi*x)


    V_plot = VectorFunctionSpace(mesh, "DG", 1)
    sigmaexact_plot = Function(V_plot).project(sigmaexact)
    uexact_plot = Function(V).project(uexact)
    normsigmaexactY = norm(sigmaexact_plot[1], norm_type="L2")
    print("norm Y Component sigmaexact =" + str(normsigmaexactY))

    fig, axes = plt.subplots()
    quiver(sigmaexact_plot, axes=axes)
    axes.set_aspect("equal")
    axes.set_title("$\sigma$ exact")
    fig.savefig("../Results/sigmaExact.png")

    file = File("../Results/ExactSol.pvd")
    file.write(sigmaexact_plot, uexact_plot)

    #sigmaexact = Function(Sigmahat).project(sigmaexact)

    #a_hybrid = (inner(sigmahat, tauhat) * dx + div(tauhat) * uhat * dx - div(sigmahat) * vhat * dx + vhat * uhat * dx
     #           + inner(tauhat, n) * lambdar * (ds_b + ds_t + ds_v)
      #          + inner(sigmahat , n) * gammar * (ds_b + ds_t + ds_v)
       #         + jump(tauhat, n=n) * lambdar('+') * (dS_h + dS_v)
        #        + jump(sigmahat, n=n) * gammar('+') * (dS_h + dS_v))

    a_hybrid_Piola = (inner(sigmahat, tauhat) * dx + div(tauhat) * uhat * dx
                - div(sigmahat) * vhat * dx + vhat * uhat * dx
                + inner(tauhat, n) * lambdar * (ds_b + ds_t + ds_v)
                + inner(sigmahat, n) * gammar * (ds_b + ds_t + ds_v)
                - inner(sigmaexact, n) * gammar * (ds_b(degree=(5, 5)) + ds_t(degree=(5, 5)) + ds_v(degree=(5, 5)))
                + jump(tauhat, n=n) * lambdar('+') * (dS_h + dS_v)
                + jump(sigmahat, n=n) * gammar('+') * (dS_h + dS_v)
                - f * vhat * dx(degree=(5, 5)))


    scpc_parameters = {"ksp_type": "preonly", "pc_type": "lu"}

    solve(lhs(a_hybrid_Piola) == rhs(a_hybrid_Piola), wh, solver_parameters = {"ksp_type": "gmres","mat_type":"matfree",
                                                  "pc_type":"python", "pc_python_type":"firedrake.SCPC",
                                                  "condensed_field":scpc_parameters,
                                                  "pc_sc_eliminate_fields":"0,1"})

    sigmah, uh, lamdah = wh.split()

    fig, axes = plt.subplots()
    quiver(sigmah, axes=axes)
    axes.set_aspect("equal")
    axes.set_title("$\sigma$ with Piola")
    fig.savefig("../Results/sigmaPiola.png")

    file = File("../Results/ResultsCompareY/Piola.pvd")
    file.write(sigmah, uh)

    normSigmahY = norm(sigmah[1], norm_type="L2")

    print("norm Y Component with Piola ="+str(normSigmahY))

    return (uh, sigmah, normSigmahY)



def SolveWithIdentity(mesh):
    #################### define Function Spaces ###########################################################################

    CG_1 = FiniteElement("CG", interval, 1)
    DG_0 = FiniteElement("DG", interval, 0)
    P1P0 = TensorProductElement(CG_1, DG_0)
    RT_horiz = HDivElement(P1P0)
    RT_horiz_broken = BrokenElement(RT_horiz)
    P0P1 = TensorProductElement(DG_0, CG_1)
    RT_vert = HDivElement(P0P1)
    RT_vert_broken = BrokenElement(RT_vert)
    full = EnrichedElement(RT_horiz_broken, RT_vert_broken)
    Sigma = FunctionSpace(mesh, full)
    remapped = WithMapping(full, "identity")
    Sigmahat = FunctionSpace(mesh, remapped)

    V = FunctionSpace(mesh, "DQ", 0)
    T = FunctionSpace(mesh, FiniteElement("HDiv Trace", mesh.ufl_cell(), degree=0))

    W_hybrid = Sigmahat * V * T

    n = FacetNormal(mesh)

    sigmahat, uhat, lambdar = TrialFunctions(W_hybrid)
    tauhat, vhat, gammar = TestFunctions(W_hybrid)

    wh = Function(W_hybrid)

    f = (1 + 4 * pi ** 2) * cos(2 * pi * x)
    sigmaexact = as_vector((-2 * pi * sin(2 * pi * x), 0))
    uexact = cos(2 * pi * x)

    a_hybrid_Identity = (inner(sigmahat, tauhat) * dx + div(tauhat) * uhat * dx
                - div(sigmahat) * vhat * dx + vhat * uhat * dx
                + inner(tauhat, n) * lambdar * (ds_b + ds_t + ds_v)
                + inner(sigmahat, n) * gammar * (ds_b + ds_t + ds_v)
                - inner(sigmaexact, n) * gammar * (ds_b(degree=(5, 5)) + ds_t(degree=(5, 5)) + ds_v(degree=(5, 5)))
                + jump(tauhat, n=n) * lambdar('+') * (dS_h + dS_v)
                + jump(sigmahat, n=n) * gammar('+') * (dS_h + dS_v)
                - f * vhat * dx(degree=(5, 5)))

    #L = -f * vhat * dx

    # will be the solution
    wh = Function(W_hybrid)

    # solve

    parameters = {"ksp_type": "gmres", "ksp_monitor": None,
                  "pc_type": "lu", "mat_type": "aij",
                  "pc_factor_mat_solver_type": "mumps"}

    solve(lhs(a_hybrid_Identity) == rhs(a_hybrid_Identity), wh, bcs=[], solver_parameters=parameters)

    sigmah, uh, lambdah = wh.split()

    fig, axes = plt.subplots()
    quiver(sigmah, axes=axes)
    axes.set_aspect("equal")
    axes.set_title("$\sigma$ with Identity")
    fig.savefig("../Results/sigmaIdentity.png")

    file2 = File("../Results/ResultsCompareY/Identity.pvd")
    file2.write(sigmah, uh)

    normSigmahY = norm(sigmah[1], norm_type="L2")
    print("norm Y Component with Identity =" + str(normSigmahY))

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

    f = (1 + 4 * pi ** 2) * cos(2 * pi * x)
    sigmaexact = as_vector((-2 * pi * sin(2 * pi * x), 0))
    uexact = cos(2 * pi * x)

    bc0 = DirichletBC(W.sub(0), sigmaexact, 1)
    bc1 = DirichletBC(W.sub(0), sigmaexact, 2)

    a_hybrid_Identity_BrokenVert = (inner(sigmahat, tauhat) * dx + div(tauhat) * uhat * dx
                - div(sigmahat) * vhat * dx + vhat * uhat * dx
                + inner(tauhat, n) * lambdar * (ds_b + ds_t)
                + inner(sigmahat, n) * gammar * (ds_b + ds_t)
                - inner(sigmaexact, n) * gammar * (ds_b(degree=(5, 5)) + ds_t(degree=(5, 5)) )
                + jump(tauhat, n=n) * lambdar('+') * (dS_h)
                + jump(sigmahat, n=n) * gammar('+') * (dS_h)
                - f * vhat * dx(degree=(5, 5)))

    #a_hybrid = (inner(sigmahat, tauhat) * dx + div(tauhat) * uhat * dx - div(sigmahat) * vhat * dx + vhat * uhat * dx
     #           + inner(tauhat, n) * lambdar * (ds_b + ds_t + ds_v)
      #          + inner(sigmahat, n) * gammar * (ds_b + ds_t + ds_v)
       #         + jump(tauhat, n=n) * lambdar('+') * (dS_h)
        #        + jump(sigmahat, n=n) * gammar('+') * (dS_h))

    #L = f * vhat * dx

    # will be the solution
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
    fig.savefig("../Results/sigmaIdBrokenVert.png")

    file2 = File("../Results/ResultsCompareY/Identity_BrokenVert.pvd")
    file2.write(sigmah, uh)

    normSigmahY = norm(sigmah[1], norm_type="L2")
    print("norm Y Component with identity, only broken in vertical =" + str(normSigmahY))

    return (uh, sigmah, normSigmahY)



########################################################################################################################

##############     define mesh ######################################################################

m = IntervalMesh(10,2)
mesh = ExtrudedMesh(m, 5, extrusion_type='uniform')

Vc = mesh.coordinates.function_space()
x, y = SpatialCoordinate(mesh)
f_mesh = Function(Vc).interpolate(as_vector([x, y + ( 0.25 * x**4 -x**3 + x**2) * (1-y) ] ) )
mesh.coordinates.assign(f_mesh)

xs = [mesh.coordinates.dat.data[i][0] for i in range(0,66)]
ys = [mesh.coordinates.dat.data[i][1] for i in range(0,66)]

plt.scatter(xs, ys)


######################################################

SolveWithPiola(mesh)

SolveWithIdentity(mesh)

SolveWithIdentity_BrokenVertical(mesh)