import matplotlib.pyplot as plt
from firedrake import *

def brokenSpace (mesh):
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

    return W_hybrid


def brokenSpace_vert (mesh, ):
    family = "CG"
    horizontal_degree = 0
    vertical_degree = 0
    S1 = fd.FiniteElement(family, fd.interval, horizontal_degree + 1)
    S2 = fd.FiniteElement("DG", fd.interval, horizontal_degree)

    # vertical base spaces
    T0 = fd.FiniteElement("CG", fd.interval, vertical_degree + 1)
    T1 = fd.FiniteElement("DG", fd.interval, vertical_degree)
    Tlinear = fd.FiniteElement("CG", fd.interval, 1)

    # build spaces V2, V3, Vt
    V2h_elt = fd.HDiv(fd.TensorProductElement(S1, T1))
    V2t_elt = fd.TensorProductElement(S2, T0)
    V3_elt = fd.TensorProductElement(S2, T1)
    V2v_elt = fd.HDiv(V2t_elt)
    V2v_elt_Broken = fd.BrokenElement(fd.HDiv(V2t_elt))
    # V2_elt = V2h_elt + V2v_elt
    V2_elt = fd.EnrichedElement(V2h_elt, V2v_elt_Broken)
    VT_elt = fd.TensorProductElement(S2, Tlinear)

    V1 = fd.FunctionSpace(mesh, V2_elt, name="HDiv")
    remapped = fd.WithMapping(V2_elt, "identity")
    V1 = fd.FunctionSpace(mesh, remapped, name="HDiv")

    V2 = fd.FunctionSpace(mesh, V3_elt, name="DG")
    Vt = fd.FunctionSpace(mesh, V2t_elt, name="Temperature")
    Vv = fd.FunctionSpace(mesh, V2v_elt, name="Vv")

    T = fd.FunctionSpace(mesh, VT_elt)

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

#compare matrix structures for a mixed Poisson problem
W = RT_Space_classic(mesh)
n = FacetNormal(mesh)
u, p = TrialFunctions(W)
v, phi= TestFunctions(W)


a_classic = (inner(u, v) * dx + div(v) * p * dx
                      - div(u) * phi * dx)
W = brokenSpace(mesh)
u, p, lambdar = TrialFunctions(W)
v, phi, gammar = TestFunctions(W)

a_broken =(inner(u, v) * dx + div(v) * p * dx - div(u) * phi * dx + phi * p * dx
            + inner(v, n) * lambdar * (ds_b + ds_t + ds_v)
            + inner(u, n) * gammar * (ds_b + ds_t + ds_v)
            + jump(v, n=n) * lambdar('+') * (dS_h + dS_v)
            + jump(u, n=n) * gammar('+') * (dS_h + dS_v))

W = brokenSpace_vert(mesh)
u, p, lambdar = TrialFunctions(W)
v, phi, gammar = TestFunctions(W)


a_broken_vert =(inner(u, v) * dx + div(v) * p * dx
                - div(u) * phi * dx
                + inner(v, n) * lambdar * (ds_b + ds_t)
                + inner(u, n) * gammar * (ds_b + ds_t)
                + jump(v, n=n) * lambdar('+') * (dS_h)
                + jump(u, n=n) * gammar('+') * (dS_h))



A_classic = assemble(a_classic)
A_broken = assemble(a_broken)
A_broken_vert = assemble(a_broken_vert)

#plot matrix structures

plt.spy(A_classic)
plt.spy(A_broken)
plt.spy(A_broken_vert)