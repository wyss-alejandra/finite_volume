import fv_rk3_tvd as f
import numpy as np

discontinuity_point = 5
value_right = 0
value_left = 1

n = 8
number_of_volumes = 200
dimensions = 1

a = 0
b = 10
h = (b - a) / number_of_volumes

z = np.zeros((number_of_volumes + 7, dimensions)).reshape(number_of_volumes + 7, dimensions)
z_start = (a - h) - 2 * h
z_stop = b + 3 * h
z = np.linspace(start=z_start, stop=z_stop, num=len(z)).reshape(len(z), dimensions)

z1 = z[3, 0]
z2 = z[4, 0]

print(z1, z2)

gs = f.gauss_sol(z1, z2, h, n, discontinuity_point=discontinuity_point, value_left=value_left, value_right=value_right,
                 return_x_w=True)
print("sol ", gs[0])
print("xr ", gs[1])
print("wr ", gs[2])
