from firedrake import *
import matplotlib.pyplot as plt

def test_HDIV_linesmoother_Star(mesh, S1family, L, H):
    nits = []
    horizontal_degree = 0
    vertical_degree = 0

    S1 = FiniteElement(S1family, mesh._base_mesh.ufl_cell(), horizontal_degree+1)
    S2 = FiniteElement("DG", mesh._base_mesh.ufl_cell(), horizontal_degree)
    T0 = FiniteElement("CG", interval, vertical_degree+1)
    T1 = FiniteElement("DG", interval, vertical_degree)


    V2h_elt = HDiv(TensorProductElement(S1, T1))
    V2t_elt = TensorProductElement(S2, T0)
    V2v_elt = HDiv(V2t_elt)
    V3_elt = TensorProductElement(S2, T1)
    V2_elt = EnrichedElement(V2h_elt, V2v_elt)

    V = FunctionSpace(mesh, V2_elt)


    ##################################################

    u = TrialFunction(V)
    v = TestFunction(V)

    a = (inner(u, v) + inner(div(u), div(v)))*dx

    bcs = [DirichletBC(V.sub(0), 0, "on_boundary"),
           DirichletBC(V.sub(0), 0, "top"),
           DirichletBC(V.sub(0), 0, "bottom")]

    x = SpatialCoordinate(mesh)
    if len(x) == 2:
        rsq = (x[0]-L/2)**2/20**2 + (x[1]-H/2)**2/0.2**2
    else:
        rsq = (x[0]-L/2)**2/20**2 + (x[1] - 50)**2/20**2 + (x[2]-H/2)**2/0.2**2
    f = as_vector([exp(-rsq),1])

    L = inner(f, v)*dx

    uh = Function(V)
    problem = LinearVariationalProblem(a, L,uh, bcs=bcs)

    sparameters = {
        "mat_type": "matfree",
        'snes_monitor': None,
        "ksp_type": "gmres",
        "ksp_monitor": None,
        'pc_type': 'python',
        "pc_python_type": "firedrake.AssembledPC",
        "assembled_ksp_type": "preonly",
        "assembled_pc_type": "python",
        'assembled_pc_python_type': 'firedrake.ASMStarPC',
        'assembled_pc_star_dims': '0',
        "assembled_pc_star_sub_pc_type": "lu",
        "assembled_pc_star_sub_sub_pc_factor_mat_solver_type": "mumps"
    }

    LU = {
        "ksp_type": "preonly",
        "pc_type": "python",
        "pc_python_type": "firedrake.AssembledPC",
        "assembled_pc_type": "lu",
        "assembled_pc_factor_mat_solver_type": "mumps"
    }

    wave_parameters = {'mat_type': 'matfree',
                       'ksp_type': 'preonly',
                       'pc_type': 'python',
                       'pc_python_type': 'firedrake.HybridizationPC',
                       'hybridization': {'ksp_type': 'cg',
                                         'ksp_monitor': None}}
    ls = {          'pc_type': 'python',
                    'pc_python_type': 'firedrake.ASMLinesmoothPC',
                    'pc_linesmooth_codims': '0, 1',
                    'pc_linesmooth_star': '1',}

    wave_parameters['hybridization'].update(ls)

    LS_solver = LinearVariationalSolver(problem, solver_parameters= sparameters)
    LS_solver.solve()

    file1 = File("../Results/Linesmoother/HdivSpace_LS.pvd")
    file1.write(uh)

    #LU_solver = LinearVariationalSolver(problem, solver_parameters=LU)
    #LU_solver.solve()

    #file2 = File("../Results/Linesmoother/HdivSpace_LU.pvd")
    #file2.write(uh)
    #ctx = solver.snes.ksp.pc.getPythonContext()
    #nits.append(ctx.trace_ksp.getIterationNumber())
    #print(ctx.trace_ksp.getIterationNumber())
    return 0
    #assert nits == expected

def test_BrokenVert_linesmoother_Star(mesh, S1family, L, H):
    nits = []
    horizontal_degree = 0
    vertical_degree = 0

    S1 = FiniteElement(S1family, mesh._base_mesh.ufl_cell(), horizontal_degree+1)
    S2 = FiniteElement("DG", mesh._base_mesh.ufl_cell(), horizontal_degree)
    T0 = FiniteElement("CG", interval, vertical_degree+1)
    T1 = FiniteElement("DG", interval, vertical_degree)

    Tlinear = FiniteElement("CG", interval, 1)
    VT_elt =TensorProductElement(S2, Tlinear)

    V2h_elt = HDiv(TensorProductElement(S1, T1))
    V2t_elt = TensorProductElement(S2, T0)
    V2v_elt = HDiv(V2t_elt)
    V2v_elt_Broken = BrokenElement(HDiv(V2t_elt))
    V3_elt = TensorProductElement(S2, T1)
    V2_elt = EnrichedElement(V2h_elt, V2v_elt_Broken)

    V = FunctionSpace(mesh, V2_elt)
    remapped = WithMapping(V2_elt, "identity")
    V = FunctionSpace(mesh, remapped, name="HDiv")

    T = FunctionSpace(mesh, VT_elt)

    W = V * T


    ##################################################

    u, lambdar = TrialFunctions(W)
    v, gammar = TestFunctions(W)

    n = FacetNormal(mesh)

    a = (inner(u, v) * dx + div(u) *div(v) * dx
        + inner(v, n) * lambdar * (ds_b + ds_t)
        + inner(u, n) * gammar * (ds_b + ds_t)
        + jump(v, n=n) * lambdar('+') * (dS_h)
        + jump(u, n=n) * gammar('+') * (dS_h))

    bcs = [DirichletBC(W.sub(0), 0, "on_boundary"),
           DirichletBC(W.sub(0), 0, "top"),
           DirichletBC(W.sub(0), 0, "bottom")]

    x = SpatialCoordinate(mesh)
    if len(x) == 2:
        rsq = (x[0] - L/2) ** 2 / 20 ** 2 + (x[1] - H/2) ** 2 / 0.2 ** 2
    else:
        rsq = (x[0] - L/2) ** 2 / 20 ** 2 + (x[1] - 50) ** 2 / 20 ** 2 + (x[2] - H/2) ** 2 / 0.2 ** 2
    f = as_vector([exp(-rsq),1])

    L = inner(f, v)*dx

    wh = Function(W)
    problem = LinearVariationalProblem(a, L, wh)

    sparameters = {
        "mat_type": "matfree",
        'snes_monitor': None,
        "ksp_type": "gmres",
        "ksp_view": None,
        "ksp_monitor": None,
        'pc_type': 'python',
        "pc_python_type": "firedrake.AssembledPC",
        "assembled_ksp_type": "preonly",
        "assembled_pc_type": "python",
        'assembled_pc_python_type': 'firedrake.ASMStarPC',
        'assembled_pc_star_dims': '0',
        "assembled_pc_star_sub_pc_type": "lu",
        'assembled_pc_star_sub_pc_factor_mat_solver_type' : 'mumps',
    }

    LU = {
        "ksp_type": "preonly",
        "pc_type": "python",
        "pc_python_type": "firedrake.AssembledPC",
        "assembled_pc_type": "lu",
        "assembled_pc_factor_mat_solver_type": "mumps"
    }


    wave_parameters = {'mat_type': 'matfree',
                       'ksp_type': 'preonly',
                       'pc_type': 'python',
                       'pc_python_type': 'firedrake.HybridizationPC',
                       'hybridization': {'ksp_type': 'cg',
                                         'ksp_monitor': None}}
    ls = {          'pc_type': 'python',
                    'pc_python_type': 'firedrake.ASMLinesmoothPC',
                    'pc_linesmooth_codims': '0, 1',
                    'pc_linesmooth_star': '1',}

    wave_parameters['hybridization'].update(ls)

    LS_solver = LinearVariationalSolver(problem, solver_parameters=sparameters)
    LS_solver.solve()

    uh, lamdah = wh.split()

    file1 = File("../Results/Linesmoother/NewSpace_LS.pvd")
    file1.write(uh)

    LU_solver = LinearVariationalSolver(problem, solver_parameters=LU)
    LU_solver.solve()

    uh, lamdah = wh.split()

    file2 = File("../Results/Linesmoother/NewSpace_LU.pvd")
    file2.write(uh)
    #ctx = solver.snes.ksp.pc.getPythonContext()
    #nits.append(ctx.trace_ksp.getIterationNumber())
    #print(ctx.trace_ksp.getIterationNumber()
    return 0

L = 3.0e5
H = 1.0e4
nlayers = 50
mesh = ExtrudedMesh(PeriodicIntervalMesh(100, L), nlayers, H / nlayers)
S1family = "CG"

#test_HDIV_linesmoother_Star(mesh, S1family, L, H)

test_BrokenVert_linesmoother_Star(mesh, S1family, L, H)
