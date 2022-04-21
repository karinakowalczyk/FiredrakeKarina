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

        #W_hydrostatic = MixedFunctionSpace((Vv, V1, T))

        # EDIT: return full spaces for full equations later

        return (V0, Vv, V1, V2, T)


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


def compressible_hydrostatic_balance(parameters, theta0, rho0, lambdar0, pi0=None,
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
    _, Vv, Vp, Vt, Vtr = build_spaces(mesh, vertical_degree=1, horizontal_degree=1) # arguments to be set in main function
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
    arhs += gammar * pi_boundary*bmeasure

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

            + gammar * (lambdar - pi_boundary) * bmeasure
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
    lambdar0.assign(lambdar)
#######################################################################################################

##set up mesh and parameters for main computations

parameters = Parameters()
g = parameters.g
c_p = parameters.cp

dT = Constant(0)

nlayers = 5
columns = 10
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

# set up fem spaces
V0, _, Vp, Vt, Vtr = build_spaces(mesh, vertical_degree=1, horizontal_degree=1)
W = V0*Vp*Vt*Vtr

# Hydrostatic case: Isothermal with T = 250, define background temperature
Tsurf = 250.
N = g/sqrt(c_p*Tsurf)

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
x,z = SpatialCoordinate(mesh)
thetab = Tsurf*exp(N**2*z/g)
theta_b = Function(Vt).interpolate(thetab)

# Calculate hydrostatic Pi and rho by solving compressible balance equation, to be used as initial guess for the
# full solver later
Pi = Function(Vp)
rho_b = Function(Vp)
lambdarb = Function(Vtr)

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

compressible_hydrostatic_balance(parameters, theta_b, rho_b, lambdarb, Pi,
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

compressible_hydrostatic_balance(parameters, theta_b, rho_b, lambdarb, Pi,
                                 top=True, params=piparamsSCPC)
p1 = minimum(Pi)
alpha = 2.*(p1-p0)
beta = p1-alpha
pi_top = (1.-beta)/alpha

print("SOLVE FOR RHO NOW")

#rho_b to be used later as initial guess for solving Euler equations
compressible_hydrostatic_balance(parameters, theta_b, rho_b, lambdarb, Pi,
                                     top=True, pi_boundary=pi_top, solve_for_rho=True,
                                     params=piparamsSCPC)


#initialise functions

theta0 = Function(Vt, name="theta0").interpolate(theta_b)
rho0 = Function(Vp, name="rho0").interpolate(rho_b) # where rho_b solves the hydrostatic balance eq.
u0 = Function(V0, name="u0").project(as_vector([20.0, 0.0]))
File_test = File("Results/compEuler/testu0.pvd")
#lambdar_guess = z-1
#lambdar0 = Function(Vtr).assign(lambdar_guess)
lambdar0 = Function(Vtr, name="lambda0").assign(lambdarb) # we use lambda from hzdrostatic solve as initial guess
File_test.write(u0, theta0, rho0, lambdar0)

zvec = as_vector([0,1])
n = FacetNormal(mesh)

########## define ttrial functions##################

Un =Function(W)
Unp1 = Function(W)

x, z = SpatialCoordinate(mesh)

un, rhon, thetan, lambdarn = Un.split()

un.assign(u0)
rhon.assign(rho0)
thetan.assign(theta0)
lambdarn.assign(lambdar0)

File_test = File("Results/compEuler/testun1.pvd")
File_test.write(un, rhon, thetan, lambdarn)

print("rho max min", rhon.dat.data.max(),  rhon.dat.data.min())
print("theta max min", thetan.dat.data.max(), thetan.dat.data.min())
print("lambda max min", lambdarn.dat.data.max(), lambdarn.dat.data.min())

#bn.interpolate(fd.sin(fd.pi*z/H)/(1+(x-xc)**2/a**2))
#bn.interpolate(fd.Constant(0.0001))


#The timestepping solver
un, rhon, thetan, lamdan = split(Un)

unp1, rhonp1, thetanp1, lamdanp1 = split(Unp1)

unph = 0.5*(un + unp1)
thetanph = 0.5*(thetan + thetanp1)
lamdanph = 0.5*(lamdan + lamdanp1)
rhonph = 0.5*(rhon + rhonp1)
#Ubar = fd.as_vector([U, 0])-
ubar = unph
n = FacetNormal(mesh)
unn = 0.5*(dot(unph, n) + abs(dot(unph, n)))

Pin = thermodynamics_pi(parameters, rhon, thetan)
Pinp1 = thermodynamics_pi(parameters, rhonp1, thetanp1)
Pinph = 0.5*(Pin + Pinp1)

################################################################

Upwind = 0.5*(sign(dot(unph, n))+1)

perp_u_upwind = lambda q: Upwind('+')*perp(q('+')) + Upwind('-')*perp(q('-'))
u_upwind = lambda q: Upwind('+')*q('+') + Upwind('-')*q('-')



def uadv_eq(w):
    return( -inner(perp(grad(inner(w, perp(unph)))), unph)*dx
                     - inner(jump(  inner(w, perp(unph)), n), perp_u_upwind(unph))*(dS)
                     #- inner(inner(w, perp(unph))* n, unph) * ( ds_t + ds_b )
                     - 0.5 * inner(unph, unph) * div(w) * dx
                     #+ 0.5 * inner(u_upwind(unph), u_upwind(unph)) * jump(w, n) * dS_h
             )
#add boundary surface terms/BC
def u_eqn(w, gammar):
    return ( inner(w, unp1 - un)*dx + dT* (uadv_eq(w) - c_p*div(w*thetanph)* Pinph*dx
                + c_p*jump(thetanph*w, n)*lamdanp1('+')*dS_h
                + c_p*inner(thetanph*w, n)*lamdanp1*(ds_t + ds_b)
                + c_p*jump(thetanph*w, n)*(0.5*(Pinph('+') + Pinph('-')))*(dS_v)
                #+ c_p * inner(thetanph * w, n) * Pinph * (ds_v)
                + gammar('+')*jump(unp1,n)*dS_h
                + gammar*inner(unp1,n)*(ds_t + ds_b)
                + g * inner(w,zvec)*dx)
                 )

#check signs everywhere
unn = 0.5*(dot(unph, n) + abs(dot(unph, n)))
#q=rho
dS = dS_h + dS_v
def rho_eqn(phi):
    return ( phi*(rhonp1 - rhon)*dx - dT * (inner(grad(phi), outer(rhonph, unph))*dx
                + dot(jump(phi,n), (un('+')*rhonph('+') - un('-')*rhonph('-')))*dS
               #+ dot(phi*unph,n) *ds_v
                    )
                )


def theta_eqn(chi):
    return (chi*(thetanp1 - thetan)*dx + dT* (inner(outer(chi, unph), grad(thetanph))*dx
                    + dot(jump(chi,n), (un('+')*thetanph('+') - un('-')*thetanph('-')))*dS
                    - (inner(chi('+'), dot(unph('+'), n('+'))*thetanph('+'))
                      + inner(chi('-'), dot(unph('-'), n('-'))*thetanph('-')))*dS

                    #+ dot(unph*chi,n)*thetanph * (ds_v + ds_t + ds_b)
                    #- inner(chi*thetanph * unph, n)* (ds_v +  ds_t + ds_b)
                 )
            )

w, phi, chi, gammar = TestFunctions(W)
gamma = Constant(10000.0)
eqn = u_eqn(w, gammar) + theta_eqn(chi) + rho_eqn(phi) # + gamma * rho_eqn(div(w))

nprob = NonlinearVariationalProblem(eqn, Unp1)


luparams = {'snes_monitor':None,
    'mat_type':'aij',
    'ksp_type':'preonly',
    'pc_type':'lu',
    'pc_factor_mat_solver_type':'mumps'}

sparameters = {
    "mat_type":"matfree",
    'snes_monitor': None,
    "snes_converged_reason": None,
    "ksp_type": "fgmres",
    "ksp_gmres_modifiedgramschmidt": None,
    "ksp_converged_reason": None,
    'ksp_monitor': None,
    "ksp_rtol": 1e-8,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_0_fields": "0,2,3",
    "pc_fieldsplit_1_fields": "1",
    "pc_fieldsplit_schur_fact_type": "full",
    "pc_fieldsplit_off_diag_use_amat": True,
}

sparameters_exact = { "mat_type": "aij",
                   'snes_monitor': None,
                   'snes_view': None,
                   #'snes_type' : 'ksponly',
                   'ksp_monitor_true_residual': None,
                   'snes_converged_reason': None,
                   'ksp_converged_reason': None,
                   "ksp_type" : "preonly",
                   "pc_type" : "lu",
                   "pc_factor_mat_solver_type": "mumps"
                   }

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
bottomright = {
    "ksp_type": "gmres",
    "ksp_max_it": 3,
    "pc_type": "python",
    "pc_python_type": "firedrake.MassInvPC",
    "Mp_pc_type": "bjacobi",
    "Mp_sub_pc_type": "ilu"
}

sparameters["fieldsplit_1"] = bottomright


sparameters["fieldsplit_0"] = topleft_LS


nsolver = NonlinearVariationalSolver(nprob, solver_parameters=sparameters_exact)

name = "Results/compEuler/full/euler_semi_imp"
file_gw = File(name+'.pvd')
un, rhon, thetan, lamdan = Un.split()
file_gw.write(un, rhon, thetan, lambdarn)
Unp1.assign(Un)

"""
name2 = "Results/compEuler/perturbations/euler_perturbations"
file2 = File(name+'.pvd')

un_pert = Function(V0).project(un - u0)
rhon_pert = Function(Vp).interpolate(rhon - rho0)
thetan_pert = Function(Vt).interpolate(thetan - theta0)
file2.write(un_pert, rhon_pert, thetan_pert)
"""

dt = 1.
dumpt = 1.
tdump = 0.
dT.assign(dt)
tmax = 13.


print('tmax', tmax, 'dt', dt)
t = 0.
while t < tmax - 0.5*dt:
    print(t)
    t += dt
    tdump += dt

    nsolver.solve()
    Un.assign(Unp1)

    if tdump > dumpt - dt*0.5:
        file_gw.write(un, rhon, thetan, lambdarn)
        #file2.write(un_pert, rhon_pert, thetan_pert)
        tdump -= dumpt


