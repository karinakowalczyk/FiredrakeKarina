import matplotlib.pyplot as plt
from firedrake import *
H=5
m = IntervalMesh(10,1)
mesh = ExtrudedMesh(m, 5, layer_height = H/5, extrusion_type='uniform')
L=1.

Vc = mesh.coordinates.function_space()
x, y = SpatialCoordinate(mesh)
new_coords = Function(Vc).interpolate(as_vector([x,y -(4/H**2) *0.3*(sin(2*pi*x/L))*((y-H/2)**2 - H**2/4)]  ))
mesh = Mesh(new_coords)
xs = [mesh.coordinates.dat.data[i][0] for i in range(0,66)]
ys = [mesh.coordinates.dat.data[i][1] for i in range(0,66)]

plt.scatter(xs, ys)
plt.show()




H = 50000.  # Height position of the model top
ext_mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
Vc = ext_mesh.coordinates.function_space()
x, y = SpatialCoordinate(ext_mesh)
new_coords = Function(Vc).interpolate(as_vector([x,y -(4/H**2) * exp(-x**2/2)*((y-H/2)**2 - H**2/4)]))
mesh = Mesh(new_coords)
#mesh=ext_mesh