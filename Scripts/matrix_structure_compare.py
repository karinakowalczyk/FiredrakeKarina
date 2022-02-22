import matplotlib.pyplot as plt
from firedrake import *
import scipy.sparse as sp


def brokenSpace (mesh):
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
    T = FunctionSpace(mesh, P0P1)

    W_hybrid = Sigmahat * V * T

    return W_hybrid


def brokenSpace_vert (mesh, ):
    family = "CG"
    horizontal_degree = 0
    vertical_degree = 0
    S1 = FiniteElement(family, interval, horizontal_degree + 1)
    S2 = FiniteElement("DG", interval, horizontal_degree)

    # vertical base spaces
    T0 = FiniteElement("CG", interval, vertical_degree + 1)
    T1 = FiniteElement("DG", interval, vertical_degree)
    Tlinear = FiniteElement("CG",interval, 1)

    # build spaces V2, V3, Vt
    V2h_elt = HDiv(TensorProductElement(S1, T1))
    V2t_elt = TensorProductElement(S2, T0)
    V3_elt = TensorProductElement(S2, T1)
    V2v_elt = HDiv(V2t_elt)
    V2v_elt_Broken = BrokenElement(HDiv(V2t_elt))
    # V2_elt = V2h_elt + V2v_elt
    V2_elt = EnrichedElement(V2h_elt, V2v_elt_Broken)
    VT_elt = TensorProductElement(S2, Tlinear)

    V1 = FunctionSpace(mesh, V2_elt, name="HDiv")
    remapped = WithMapping(V2_elt, "identity")
    V1 = FunctionSpace(mesh, remapped, name="HDiv")

    V2 = FunctionSpace(mesh, V3_elt, name="DG")
    Vt = FunctionSpace(mesh, V2t_elt, name="Temperature")
    Vv = FunctionSpace(mesh, V2v_elt, name="Vv")

    T = FunctionSpace(mesh, VT_elt)

    W = V1 * V2 * T  # velocity, pressure, temperature, trace of velocity

    return W

def RT_Space_classic(mesh):
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

    W = Sigma * VD

    return W


m = IntervalMesh(4,1)
mesh = ExtrudedMesh(m, 4, extrusion_type='uniform')

Vc = mesh.coordinates.function_space()
x, y = SpatialCoordinate(mesh)
#f_mesh = Function(Vc).interpolate(as_vector([x, y + ( 0.25 * x**4 -x**3 + x**2) * (1-y) ] ) )
f_mesh = Function(Vc).interpolate(as_vector([x,y - exp(-x**2/2)*((y-0.5)**2 -0.25)] ) )
#mesh.coordinates.assign(f_mesh)

#compare matrix structures for a mixed Poisson problem
W = RT_Space_classic(mesh)
n = FacetNormal(mesh)
u, p = TrialFunctions(W)
v, phi= TestFunctions(W)


a_classic = (inner(u, v) * dx + div(v) * p * dx
                      - div(u) * phi * dx + p*phi*dx)
W = brokenSpace(mesh)
u, p, lambdar = TrialFunctions(W)
v, phi, gammar = TestFunctions(W)

a_broken =(inner(u, v) * dx + div(v) * p * dx - div(u) * phi * dx + p*phi*dx
            + inner(v, n) * lambdar * (ds_b + ds_t + ds_v)
            + inner(u, n) * gammar * (ds_b + ds_t + ds_v)
            + jump(v, n=n) * lambdar('+') * (dS_h + dS_v)
            + jump(u, n=n) * gammar('+') * (dS_h + dS_v))

W = brokenSpace_vert(mesh)
u, p, lambdar = TrialFunctions(W)
v, phi, gammar = TestFunctions(W)


a_broken_vert =(inner(u, v) * dx + div(v) * p * dx
                - div(u) * phi * dx + p*phi*dx
                + inner(v, n) * lambdar * (ds_b + ds_t)
                + inner(u, n) * gammar * (ds_b + ds_t)
                + jump(v, n=n) * lambdar('+') * (dS_h)
                + jump(u, n=n) * gammar('+') * (dS_h))



A_classic = assemble(a_classic).M.handle
A_broken = assemble(a_broken).M.handle
A_broken_vert = assemble(a_broken_vert).M.handle

#plot matrix structures


fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
indptr, indices, data = A_classic.getValuesCSR()
A_classic_sp = sp.csr_matrix((data, indices, indptr), shape=A_classic.getSize())
ax0.spy(A_classic_sp)

indptr, indices, data = A_broken.getValuesCSR()
A_broken_sp = sp.csr_matrix((data, indices, indptr), shape=A_broken.getSize())
ax1.spy(A_broken_sp)


indptr, indices, data = A_broken_vert.getValuesCSR()
A_broken_vert_sp= sp.csr_matrix((data, indices, indptr), shape=A_broken_vert.getSize())
ax2.spy(A_broken_vert_sp)
plt.show()
