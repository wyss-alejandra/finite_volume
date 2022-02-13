import fv_rk3_tvd as f

dimensions = 1
value_left = 1
value_right = 0
discontinuity_point = 5
i_order = 2
number_of_volumes = 50
total_number_of_volumes = number_of_volumes + 6  # Total number of control volumes - NVol
a = 0
b = 10
final_time = 2
co = 0.5
alpha = 0.5

a1, a2, x_fig, u_fig = f.fv_rk3(i_order, number_of_volumes, a, b, final_time, co, alpha,
                                discontinuity_point, value_right, value_left)
