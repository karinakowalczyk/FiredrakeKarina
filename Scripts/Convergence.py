import matplotlib.pyplot as plt
from firedrake import *
#import petsc4py.PETSc as PETSc
#PETSc.Sys.popErrorHandler()


def SolveHelmholtzIdentityHybrid (NumberNodesX, NumberNodesY):


    ################### mesh #####################################################################

    m = IntervalMesh(NumberNodesX,2)
    mesh = ExtrudedMesh(m, NumberNodesY, extrusion_type='uniform')

    Vc = mesh.coordinates.function_space()
    x, y = SpatialCoordinate(mesh)
    f = Function(Vc).interpolate(as_vector([x, y + ( 0.25 * x**4 -x**3 + x**2) * (1-y) ] ) )
    mesh.coordinates.assign(f)

    ############## function spaces ################################################################

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

    #f = 10 * exp(-100 * ((x - 1) ** 2 + (y - 0.5) ** 2))

    uexact = cos(2* pi * x) * cos(2 * pi * y)
    sigmaexact = -2* pi * as_vector((sin(2*pi*x)*cos(2*pi*y), cos(2*pi*x)*sin(2*pi*y)))
    f = (1 + 8*pi*pi) * uexact

    a_hybrid = (inner(sigmahat, tauhat) * dx + div(tauhat) * uhat * dx
                - div(sigmahat) * vhat * dx + vhat * uhat * dx
                + inner(tauhat, n) * lambdar * (ds_b + ds_t + ds_v)
                + inner(sigmahat, n) * gammar * (ds_b + ds_t + ds_v)
                - inner(sigmaexact, n) * gammar * (ds_b(degree = (5,5)) + ds_t(degree = (5,5)) + ds_v(degree = (5,5)))
                + jump(tauhat, n=n) * lambdar('+') * (dS_h + dS_v)
                + jump(sigmahat, n=n) * gammar('+') * (dS_h + dS_v)
                - f * vhat * dx(degree = (5,5)) )



    ######################### solve ###############################################################

    #will be the solution
    wh = Function(W_hybrid)

    #solve

    scpc_parameters = {"ksp_type":"preonly", "pc_type":"lu"}

    solve(lhs(a_hybrid) == rhs(a_hybrid), wh, solver_parameters = {"ksp_type": "gmres","mat_type":"matfree",
                                                  "pc_type":"python", "pc_python_type":"firedrake.SCPC",
                                                  "condensed_field":scpc_parameters,
                                                  "pc_sc_eliminate_fields":"0,1"})

    sigmah, uh, lamdah = wh.split()

    sigmaexact = Function(Sigmahat, name = "sigmaexact").project(sigmaexact)
    uexact = Function(V, name = "Uexact").project(uexact)

    #file2 = File("Conv.pvd")
    #file2.write(sigmah, sigmaexact, uh, uexact)

    fig, axes = plt.subplots()
    quiver(sigmah, axes=axes)
    axes.set_aspect("equal")
    axes.set_title("$\sigma$")
    fig.savefig("../Results/sigma_"+str(NumberNodesY)+".png")

    return (uh, mesh, V, Sigmahat)




def SolveHelmholtzIdentityHybridBrokenVert (NumberNodesX, NumberNodesY):


    ################### mesh #####################################################################

    m = IntervalMesh(NumberNodesX,2)
    mesh = ExtrudedMesh(m, NumberNodesY, extrusion_type='uniform')

    Vc = mesh.coordinates.function_space()
    x, y = SpatialCoordinate(mesh)
    f = Function(Vc).interpolate(as_vector([x, y + ( 0.25 * x**4 -x**3 + x**2) * (1-y) ] ) )
    mesh.coordinates.assign(f)

    ############## function spaces ################################################################

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
    #T = FunctionSpace(mesh, FiniteElement("HDiv Trace", mesh.ufl_cell(), degree=0))

    W_hybrid = Sigmahat * V * T

    n = FacetNormal(mesh)

    sigmahat, uhat, lambdar = TrialFunctions(W_hybrid)
    tauhat, vhat, gammar = TestFunctions(W_hybrid)

    wh = Function(W_hybrid)

    #f = 10 * exp(-100 * ((x - 1) ** 2 + (y - 0.5) ** 2))

    uexact = cos(2* pi * x) * cos(2 * pi * y)
    sigmaexact = -2* pi * as_vector((sin(2*pi*x)*cos(2*pi*y), cos(2*pi*x)*sin(2*pi*y)))
    f = (1 + 8*pi*pi) * uexact

    bc0 = DirichletBC(W_hybrid.sub(0), sigmaexact, 1)
    bc1 = DirichletBC(W_hybrid.sub(0), sigmaexact, 2)

    a_hybrid_Identity_BrokenVert = (inner(sigmahat, tauhat) * dx + div(tauhat) * uhat * dx
                                    - div(sigmahat) * vhat * dx + vhat * uhat * dx
                                    + inner(tauhat, n) * lambdar * (ds_b + ds_t)
                                    + inner(sigmahat, n) * gammar * (ds_b + ds_t)
                                    - inner(sigmaexact, n) * gammar * (ds_b(degree=(5, 5)) + ds_t(degree=(5, 5)))
                                    + jump(tauhat, n=n) * lambdar('+') * (dS_h)
                                    + jump(sigmahat, n=n) * gammar('+') * (dS_h)
                                    - f * vhat * dx(degree=(5, 5)))



    ######################### solve ###############################################################

    #will be the solution
    wh = Function(W_hybrid)

    #solve

    scpc_parameters = {"ksp_type": "preonly", "pc_type": "lu"}
    parameters = {"pc_type": "lu", "pc_factor_mat_solver_type": "mumps", "ksp_type": "preonly"}

    solve(lhs(a_hybrid_Identity_BrokenVert) == rhs(a_hybrid_Identity_BrokenVert), wh, solver_parameters=parameters, bcs=[bc0, bc1])
    sigmah, uh, lamdah = wh.split()

    sigmaexact = Function(Sigmahat, name = "sigmaexact").project(sigmaexact)
    uexact = Function(V, name = "Uexact").project(uexact)

    #file2 = File("Conv.pvd")
    #file2.write(sigmah, sigmaexact, uh, uexact)

    fig, axes = plt.subplots()
    quiver(sigmah, axes=axes)
    axes.set_aspect("equal")
    axes.set_title("$\sigma$")
    fig.savefig("../Results/sigma_"+str(NumberNodesY)+".png")

    return (uh, mesh, V, Sigmahat)



####################################################################################################################



NumberX = 10
NumberY = 5

meshSizeList = []
errorsL2Proj = []
errorsL2 = []


for count in range(0,7):

    u_curr, mesh_curr, V_curr, Sigma_curr = SolveHelmholtzIdentityHybridBrokenVert(NumberX, NumberY)

    x, y = SpatialCoordinate(mesh_curr)
    uexact = cos(2 * pi * x) * cos(2 * pi * y)
    sigmaexact = -2 * pi * as_vector((sin(2 * pi * x) * cos(2 * pi * y), cos(2 * pi * x) * sin(2 * pi * y)))

    errorUL2 = norm(uexact - u_curr, norm_type="L2")

    sigmaexact = Function(Sigma_curr, name="sigmaexact").project(sigmaexact)
    uexact = Function(V_curr, name="Uexact").project(uexact)

    differenceUProj =  uexact - u_curr
    errorUL2Proj = norm(differenceUProj, norm_type="L2")
   # errorSig = assemble((Sigma_curr - sigmaexact)**2 * dx)

    h = 1/NumberY
    meshSizeList.append(h)
    errorsL2Proj.append(errorUL2Proj)
    errorsL2.append(errorUL2)

    NumberX *=2
    NumberY *=2
    print(h)

######################################### results ###############################################################

def square (h):
    return h**2

hsquared = list(map(square, meshSizeList))

fig, axes = plt.subplots()
axes.set_title("errors")
plt.loglog(meshSizeList, errorsL2, axes = axes, color = "green", label = "L2 error", marker = ".")
plt.loglog(meshSizeList, errorsL2Proj, axes = axes, color = "blue", label = "L2 error (project uexact first)", marker = ".")
plt.loglog(meshSizeList, hsquared, axes = axes, color = "orange", label = "h^2", marker = ".")
plt.loglog(meshSizeList, meshSizeList, axes = axes, color = "red", label = "h", marker = ".")
axes.legend()
fig.savefig("../Results/errors.png")
plt.show()

