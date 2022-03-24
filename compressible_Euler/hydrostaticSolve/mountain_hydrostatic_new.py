from firedrake import *
import sys

'''
    version of "mountain_hydrostatic in gusto for new velocity space
    without making use of gusto
'''
#set physical parameters

class Parameters:
    N = 0.01  # Brunt-Vaisala frequency (1/s)
    cp = 1004.5  # SHC of dry air at const. pressure (J/kg/K)
    R_d = 287.  # Gas constant for dry air (J/kg/K)
    kappa = 2.0/7.0  # R_d/c_p
    p_0 = 1000.0*100.0  # reference pressure (Pa, not hPa)
    cp = 1004.
    g = 9.80665

def build_spaces(mesh, vertical_degree, horizontal_degree):

    if vertical_degree is not None:
        # horizontal base spaces
        cell = mesh._base_mesh.ufl_cell().cellname()
        S1 = FiniteElement("CG", cell, horizontal_degree + 1)  # EDIT: family replaced by CG (was called with RT before)
        S2 = FiniteElement("DG", cell, horizontal_degree, variant="equispaced")

        # vertical base spaces
        T0 = FiniteElement("CG", interval, vertical_degree + 1, variant="equispaced")
        T1 = FiniteElement("DG", interval, vertical_degree, variant="equispaced")

        # trace base space
        Tlinear = FiniteElement("CG", interval, 1)

        # build spaces V2, V3, Vt
        V2h_elt = HDiv(TensorProductElement(S1, T1))
        V2t_elt = TensorProductElement(S2, T0)
        V3_elt = TensorProductElement(S2, T1)
        V2v_elt = HDiv(V2t_elt)
        V2_elt = V2h_elt + V2v_elt

        V2v_elt_Broken = BrokenElement(HDiv(V2t_elt))
        V2_elt = EnrichedElement(V2h_elt, V2v_elt_Broken)
        VT_elt = TensorProductElement(S2, Tlinear)

        remapped = WithMapping(V2_elt, "identity")

        V0 = FunctionSpace(mesh, remapped, name="new_velocity")
        V1 = FunctionSpace(mesh, V3_elt, name="DG") # pressure space
        V2 = FunctionSpace(mesh, V2t_elt, name="Temp")

        T = FunctionSpace(mesh, VT_elt, name = "Trace")

        remapped = WithMapping(V2v_elt_Broken, "identity") # only test with vertical part, drop Piola transformations

        Vv = FunctionSpace(mesh, remapped, name="Vv")

        DG1_hori_elt = FiniteElement("DG", cell, 1, variant="equispaced")
        DG1_vert_elt = FiniteElement("DG", interval, 1, variant="equispaced")
        DG1_elt = TensorProductElement(DG1_hori_elt, DG1_vert_elt)
        DG1_space = FunctionSpace(mesh, DG1_elt, name = "DG1")

        W_hydrostatic = MixedFunctionSpace((Vv, V1, T))

        # EDIT: return full spaces for full equations later

        return (Vv, V1, V2, T)


def thermodynamics_pi(parameters, rho, theta_v):
    """
    Returns an expression for the Exner pressure.

    :arg parameters: a CompressibleParameters object.
    :arg rho: the dry density of air in kg / m^3.
    :arg theta: the potential temperature (or the virtual
                potential temperature for wet air), in K.
    """

    kappa = parameters.kappa
    p_0 = parameters.p_0
    R_d = parameters.R_d

    return (rho * R_d * theta_v / p_0) ** (kappa / (1 - kappa))

def thermodynamics_rho(parameters, theta_v, pi):
    """
    Returns an expression for the dry density rho in kg / m^3
    from the (virtual) potential temperature and Exner pressure.

    :arg parameters: a CompressibleParameters object.
    :arg theta_v: the virtual potential temperature in K.
    :arg pi: the Exner pressure.
    """

    kappa = parameters.kappa
    p_0 = parameters.p_0
    R_d = parameters.R_d

    return p_0 * pi ** (1 / kappa - 1) / (R_d * theta_v)


def compressible_hydrostatic_balance(parameters, theta0, rho0, pi0=None,
                                     top=False, pi_boundary=Constant(1.0),
                                     water_t=None,
                                     solve_for_rho=False,
                                     params=None):
    """
    Compute a hydrostatically balanced density given a potential temperature
    profile. By default, this uses a vertically-oriented hybridization
    procedure for solving the resulting discrete systems.

    :arg state: The :class:`State` object.
    :arg theta0: :class:`.Function`containing the potential temperature.
    :arg rho0: :class:`.Function` to write the initial density into.
    :arg top: If True, set a boundary condition at the top. Otherwise, set
    it at the bottom.
    :arg pi_boundary: a field or expression to use as boundary data for pi on
    the top or bottom as specified.
    :arg water_t: the initial total water mixing ratio field.
    """

    # Calculate hydrostatic Pi
    #VDG = state.spaces("DG")
    #Vv = state.spaces("Vv")
    #Vtr= state.spaces("Trace")
    Vv, Vp, Vt, Vtr = build_spaces(mesh, vertical_degree=1, horizontal_degree=1) # arguments to be set in main function
    W = MixedFunctionSpace((Vv, Vp, Vtr))
    v, pi, lambdar = TrialFunctions(W)
    dv, dpi, gammar = TestFunctions(W)

    n = FacetNormal(mesh)

    # add effect of density of water upon theta
    theta = theta0

    if water_t is not None:
        theta = theta0 / (1 + water_t)

    if top:
        bmeasure = ds_t
        bstring = "bottom"
        vmeasure = ds_b
    else:
        bmeasure = ds_b
        vmeasure = ds_t
        bstring = "top"

    cp = parameters.cp

    alhs = (
        (cp*inner(v, dv) - cp*div(dv*theta)*pi)*dx
        +cp * dpi*div(theta*v)*dx

        - cp*inner(theta*v, n) * gammar * vmeasure
        - cp*jump(theta*v, n=n) * gammar('+') * (dS_h)

        + cp*inner(theta*dv, n) * lambdar * vmeasure
        + cp*jump(theta*dv, n=n) * lambdar('+') * (dS_h)

        + gammar * lambdar * bmeasure
    )

    arhs = -cp*inner(dv, n)*theta*pi_boundary*bmeasure

    # Possibly make g vary with spatial coordinates?


    dim = mesh.topological_dimension()
    kvec = [0.0] * dim
    kvec[dim - 1] = 1.0
    k = Constant(kvec)

    g = parameters.g
    arhs -= g*inner(dv, k)*dx

    #bcs = [DirichletBC(W.sub(0), zero(), bstring)]
    bcs =[]

    w = Function(W)
    PiProblem = LinearVariationalProblem(alhs, arhs, w, bcs=bcs)

    if params is None:
        #params = {'ksp_type': 'preonly',
         #         'pc_type': 'python',
          #        'mat_type': 'matfree',
           #       'pc_python_type': 'gusto.VerticalHybridizationPC', #EDIT: Use SCPC instead
            #      # Vertical trace system is only coupled vertically in columns
             #     # block ILU is a direct solver!
              #    'vert_hybridization': {'ksp_type': 'preonly',
               #                          'pc_type': 'bjacobi',
                #                         'sub_pc_type': 'ilu'}}

        scpc_parameters = {"ksp_type": "preonly", "pc_type": "lu"}
        params = {"ksp_type": "gmres",
                  "snes_monitor": None,
                  "ksp_monitor": None,
                                  "mat_type": "matfree",
                                  "pc_type": "python",
                                  "pc_python_type": "firedrake.SCPC",
                                  "condensed_field": scpc_parameters,
                                  "pc_sc_eliminate_fields": "0,1"}


    PiSolver = LinearVariationalSolver(PiProblem,
                                       solver_parameters=params,
                                       options_prefix="pisolver")

    PiSolver.solve()

    v, Pi, lambdar = w.split()

    print("Pi max and min = ", Pi.dat.data.max(), Pi.dat.data.min())
    print("theta0 max and min:", theta0.dat.data.max(), theta0.dat.data.min())

    if pi0 is not None:
        pi0.assign(Pi)

    if solve_for_rho:
        w1 = Function(W)
        v, rho, lambdar = w1.split()
        rho.interpolate(thermodynamics_rho(parameters, theta0, Pi))
        print("rho max and min before", rho.dat.data.max(), rho.dat.data.min())

        v, rho, lambdar = split(w1)


        dv, dpi, gammar = TestFunctions(W)
        pi = thermodynamics_pi(parameters, rho, theta0)
        F = (
            (cp*inner(v, dv) - cp*div(dv*theta)*pi)*dx
            + cp * inner(theta * dv, n) * lambdar * vmeasure
            + cp * jump(theta * dv, n=n) * lambdar('+') * (dS_h)

            - cp * inner(theta*v, n) * gammar * vmeasure
            - cp * jump(theta*v, n=n) * gammar('+') * (dS_h)

            + dpi*div(theta0*v)*dx
            + cp*inner(dv, n)*theta*pi_boundary*bmeasure

            + gammar * lambdar * bmeasure
        )
        F += g*inner(dv, k)*dx
        rhoproblem = NonlinearVariationalProblem(F, w1, bcs=bcs)
        rhosolver = NonlinearVariationalSolver(rhoproblem, solver_parameters=params,
                                               options_prefix="rhosolver")
        rhosolver.solve()
        v, rho_, lambdar = w1.split()
        rho0.assign(rho_)
        print("rho max", rho0.dat.data.max())
    else:
        rho0.interpolate(thermodynamics_rho(parameters, theta0, Pi))

#######################################################################################################

parameters = Parameters()

dt = 5.0

nlayers = 100
columns = 200
L = 240000.
m = PeriodicIntervalMesh(columns, L)

# build volume mesh
H = 50000.  # Height position of the model top
ext_mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
Vc = VectorFunctionSpace(ext_mesh, "DG", 2)
coord = SpatialCoordinate(ext_mesh)
x = Function(Vc).interpolate(as_vector([coord[0], coord[1]]))
a = 10000.
xc = L/2.
x, z = SpatialCoordinate(ext_mesh)
hm = 1.
zs = hm*a**2/((x-xc)**2 + a**2)

xexpr = as_vector([x, z + ((H-z)/H)*zs])

new_coords = Function(Vc).interpolate(xexpr)
mesh = Mesh(new_coords)

# sponge function
W_DG = FunctionSpace(mesh, "DG", 2)
x, z = SpatialCoordinate(mesh)
zc = H-20000.
mubar = 0.3/dt
mu_top = conditional(z <= zc, 0.0, mubar*sin((pi/2.)*(z-zc)/(H-zc))**2)
mu = Function(W_DG).interpolate(mu_top)

g = parameters.g
c_p = parameters.cp

Vv, Vp, Vt, Vtr = build_spaces(mesh, vertical_degree=1, horizontal_degree=1)
# Hydrostatic case: Isothermal with T = 250
Tsurf = 250.
N = g/sqrt(c_p*Tsurf)

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
thetab = Tsurf*exp(N**2*z/g)
theta_b = Function(Vt).interpolate(thetab)

# Calculate hydrostatic Pi
Pi = Function(Vp)
rho_b = Function(Vp)

## specify solver parameters
scpc_parameters = {"ksp_type": "preonly", "pc_type": "lu"}
piparamsSCPC = {"ksp_type": "gmres",
                "ksp_monitor": None,
                #"ksp_view":None,
                "mat_type": "matfree",
                #'pc_type':'lu',
                #'pc_factor_mat_solver_type':'mumps'}
                "pc_type": "python",
                "pc_python_type": "firedrake.SCPC",
                "condensed_field": scpc_parameters,
                "pc_sc_eliminate_fields": "0,1"}

piparams_exact = {"ksp_type": "preonly",
                  "ksp_monitor": None,
                  #"ksp_view":None,
                  'pc_type':'lu',
                  'pc_factor_mat_solver_type':'mumps'
                  }

compressible_hydrostatic_balance(parameters, theta_b, rho_b, Pi,
                                 top=True, pi_boundary=0.5,
                                 params=piparamsSCPC)


def minimum(f):
    fmin = op2.Global(1, [1000], dtype=float)
    op2.par_loop(op2.Kernel("""
static void minify(double *a, double *b) {
    a[0] = a[0] > fabs(b[0]) ? fabs(b[0]) : a[0];
}
        """, "minify"), f.dof_dset.set, fmin(op2.MIN), f.dat(op2.READ))
    return fmin.data[0]

def maximum(f):
    fmax = op2.Global(1, [1000], dtype=float)
    op2.par_loop(op2.Kernel("""
static void minify(double *a, double *b) {
    a[0] = a[0] > fabs(b[0]) ? fabs(b[0]) : a[0];
}
        """, "minify"), f.dof_dset.set, fmax(op2.MAX), f.dat(op2.READ))
    return fmax.data[0]


p0 = minimum(Pi)

compressible_hydrostatic_balance(parameters, theta_b, rho_b, Pi,
                                 top=True, params=piparamsSCPC)
p1 = minimum(Pi)
alpha = 2.*(p1-p0)
beta = p1-alpha
pi_top = (1.-beta)/alpha

print("SOLVE FOR RHO NOW")

#rho_b to be used later as initial guess for solving Euler equations
compressible_hydrostatic_balance(parameters, theta_b, rho_b, Pi,
                                     top=True, pi_boundary=pi_top, solve_for_rho=True,
                                     params=piparamsSCPC)


theta0 = Function(Vt).interpolate(theta_b)
rho0 = Function(Vp).interpolate(rho_b)
u0 = Function(Vv).project(as_vector([20.0, 0.0]))

def remove_initial_w(u, Vv):
    bc = DirichletBC(u.function_space()[0], 0.0, "bottom")
    bc.apply(u)
    uv = Function(Vv).project(u)
    ustar = Function(u.function_space()).project(uv)
    uin = Function(u.function_space()).assign(u - ustar)
    u.assign(uin)

#remove_initial_w(u0, Vv)
zvec = as_vector([0,1])
n = FacetNormal(self.state.mesh)
Upwind = 0.5*(sign(dot(self.ubar, n))+1)
ubar = 0.5 (un+unp1)
uadv_eq(w, ubar) = ( -inner(perp(grad(inner(w, perp(ubar)))), q)*dx
                     - inner(jump(  inner(w, perp(ubar)), n), perp_u_upwind(q))*dS
                   )
#add boundary surface terms/BC
ueqn(w, ubar) = (uadv_eq(w,ubar) - div(w*theta)* Pi*dx \
                + jump(theta*w, n)*lambdar*dS_h # add boundary terms
                + jump(theta*w, n)*Pinph*dS_v
                + gammar*jump(u,n)*dS_h # add boundary terms
                +g* inner(w,zvec)*dx
                 )

#check signs everywhere
unn = 0.5*(dot(self.ubar, n) + abs(dot(self.ubar, n)))
#q=rho
rho_eqn(phi) = (-inner(grad(self.test), outer(q, self.ubar))*dx
                + dot(jump(self.test), (un('+')*q('+')
                                       - un('-')*q('-')))*self.dS)

#q=theta
theta_eqn(xi) = (

inner(outer(self.test, self.ubar), grad(q))*dx
+= dot(jump(self.test), (un('+')*q('+')
                                       - un('-')*q('-')))*self.dS#-= (inner(self.test('+'),
                            dot(self.ubar('+'), n('+'))*q('+'))
                      + inner(self.test('-'),
                              dot(self.ubar('-'), n('-'))*q('-')))*self.dS
)
#rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")

