import fv_rk3_tvd as f

value_left = 1
value_right = 0
discontinuity_point = 5

i_order = 2  # Type of spatial reconstruction: 0 pwc, 1 pwl, 2 WENO5

number_of_control_volumes = 200  # N

# Domain borders [a, b]
a = 0
b = 10

# Output time
final_time = 2  # TFIN

# Stability parameters: Courant (Co), diffusion (alpha)
co = 0.5
alpha = 0.5
# Call main programme.

f.fv_rk3(i_order, number_of_control_volumes, a, b, final_time, co, alpha, dimensions=1)




