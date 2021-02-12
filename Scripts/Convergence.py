import matplotlib.pyplot as plt
from firedrake import *
import petsc4py.PETSc as PETSc
PETSc.Sys.popErrorHandler()


def SolveHelmholtzIdentityHybrid (NumberNodesX, NumberNodesY):


    ################### mesh #####################################################################

    m = IntervalMesh(NumberNodesX,2)
    mesh = ExtrudedMesh(m, NumberNodesY, extrusion_type='uniform')

    Vc = mesh.coordinates.function_space()
    x, y = SpatialCoordinate(mesh)
    f = Function(Vc).interpolate(as_vector([x, y + ( 0.25 * x**4 -x**3 + x**2) * (1-y) ] ) )
    mesh.coordinates.assign(f)

    ############## function spaces ################################################################

    element = FiniteElement("RTCF", cell="quadrilateral", degree=1)
    element._mapping = 'identity'
    Sigma = FunctionSpace(mesh, element)
    V = FunctionSpace(mesh, "DG", 0)

    ########################## set up problem ####################################################

    Sigmahat = FunctionSpace(mesh, BrokenElement(Sigma.ufl_element()))  # do I need broken element here??
    V = FunctionSpace(mesh, V.ufl_element())
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


######################################################################################################################



NumberX = 10
NumberY = 5

meshSizeList = []
errorsL2 = []
errorsH1 = []
errorsL2_scaled_squared = []
errorsL2_scaled_linear = []
errorsH1_scaled = []



for count in range(0,8):
    u_curr, mesh_curr, V_curr, Sigma_curr = SolveHelmholtzIdentityHybrid(NumberX, NumberY)

    x, y = SpatialCoordinate(mesh_curr)
    uexact = cos(2 * pi * x) * cos(2 * pi * y)
    sigmaexact = -2 * pi * as_vector((sin(2 * pi * x) * cos(2 * pi * y), cos(2 * pi * x) * sin(2 * pi * y)))

    sigmaexact = Function(Sigma_curr, name="sigmaexact").project(sigmaexact)
    uexact = Function(V_curr, name="Uexact").project(uexact)

    differenceU =  uexact - u_curr
    errorUL2 = norm(differenceU, norm_type="L2")
    errorUH1 = norm(differenceU, norm_type="H1")

    h = 1/NumberY

    meshSizeList.append(h)
    errorsL2.append(errorUL2)
    errorsH1.append(errorUH1)
    errorsL2_scaled_squared.append(errorUL2/h**2)
    errorsL2_scaled_linear.append(errorUL2/h)
    NumberX *=2
    NumberY *=2
    print(h)

fig, axes = plt.subplots()
axes.set_title("errors")
plt.scatter(meshSizeList, errorsL2, axes = axes, color = "blue", label = "L2 error")
plt.plot(meshSizeList, errorsL2, color = "blue")
plt.scatter(meshSizeList, errorsH1, axes = axes, color = "orange", label = "H1 error")
plt.plot(meshSizeList, errorsH1, color = "orange")
axes.legend()
fig.savefig("../Results/errors.png")


fig, axes = plt.subplots()
axes.set_title("errors scaled")
plt.scatter(meshSizeList, errorsL2_scaled_linear, axes = axes, color = "blue", label = "L2 error/h")
plt.plot(meshSizeList, errorsL2_scaled_linear, color = "blue")
plt.scatter(meshSizeList, errorsL2_scaled_squared, axes = axes, color = "orange", label = "L2 error/h^2")
plt.plot(meshSizeList, errorsL2_scaled_squared, color = "orange")
axes.legend()
fig.savefig("../Results/errorsScaled.png")
plt.show()

#fig, axes = plt.subplots()
#axes.set_title("errors/h^2")
#plt.scatter(meshSizeList, errorsL2/(h*h), axes = axes, color = "blue", label = "L2 error")
#plt.plot(meshSizeList, errorsL2)
#plt.scatter(meshSizeList, errorsH1/(h*h), axes = axes, color = "orange", label = "H1 error")
#plt.plot(meshSizeList, errorsH1)
#plt.show()
