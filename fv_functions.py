import math

import numpy as np


def uinic(x, discontinuity_point, value_left, value_right):

    if (x >= 0) & (x <= discontinuity_point):
        value_ = value_left
    else:
        value_ = value_right

    return value_


def gauss_legendre(h, n=96, a=-1.0, b=1.0):
    """Function to obtain the Gauss-Legendre quadrature roots and weights
    :param h: h = (b - a) / n_volumes
    :param n: nth polynomial
    :param a: range starting point [a, b]
    :param b: range end point [a, b]
    :return:
    x_: roots of the nth legendre polynomial, transformed to range [a, b]
    w_: quadrature weights
    """

    assert(n >= 1)  # Number of sample points and weights must be >= 1
    # Get the roots and quadrature weights
    x, w = np.polynomial.legendre.leggauss(deg=n)

    x_ = np.zeros_like(x)
    w_ = np.zeros_like(w)

    # Transform to [a, b] range
    for i in range(0, n):
        x_[i] = (a + b) / 2 + h / 2 * x[i]
        w_[i] = h / 2 * w[i]

    return x_, w_


def gauss_sol(z1, z2, h, n, discontinuity_point, value_left, value_right, return_x_w=False):
    x, w = gauss_legendre(h=h, n=n, a=z1, b=z2)

    sol = 0
    for i in range(0, n):
        sol = sol + w[i] * uinic(x[i],
                                 discontinuity_point=discontinuity_point,
                                 value_left=value_left, value_right=value_right)

    sol = sol / h

    if return_x_w:
        return sol, x, w
    else:
        return sol


def physical_data_k(x, constant_=True):
    if constant_:
        return 0.01e-20
    else:
        return x  # Function of x that is not currently defined


def physical_data_v(x, constant_=True):
    if constant_:
        return 1
    else:
        return 0.5 + abs(math.cos(x)) / 2


def physical_data_q(x, constant_=True):
    if constant_:
        return 1
    else:
        return x  # Function of x that is not currently defined


def u_d_u_func(alpha, omega, d_, u, d_u, eps, beta):
    alpha_sum, sum_, sum_1 = 0, 0, 0
    u, d_u = np.nan, np.nan

    for j in range(0, 2):
        alpha[j] = d_[j] / ((eps + beta[j]) ** 2)
        alpha_sum += alpha[j]
    for j in range(0, 2):
        omega[j] = alpha[j] / alpha_sum
    for j in range(0, 2):
        sum_ += omega[j] * u[j]
        sum_1 += omega[j] * d_u[j]

    u_output, d_u_output = sum_, sum_1

    return u_output, d_u_output


def alpha_omega_func(alpha_, omega_, d_s, eps, beta):
    alpha_sum = 0
    for j in range(0, 2):
        alpha_[j] = d_s[j] / ((eps + beta[j]) ** 2)
        alpha_sum += alpha_[j]
    for j in range(0, 2):
        omega_[j] = alpha_[j] / alpha_sum

    return alpha_, omega_

def recons(i_order, total_volumes, u, h, x, z):
    u_r_output, u_l_output = np.zeros(total_volumes), np.zeros(total_volumes)
    d_u_r_output, d_u_l_output = np.zeros(total_volumes), np.zeros(total_volumes)

    d_r, d_l = np.zeros(3), np.zeros(3)
    d_u_r, d_u_l = np.zeros(3), np.zeros(3)

    u_z, d_u_z = np.zeros(total_volumes), np.zeros(total_volumes)
    u_l, u_r = np.zeros(3), np.zeros(3)

    alpha_r, alpha_l = np.zeros(3), np.zeros(3)
    omega_r, omega_l = np.zeros(3), np.zeros(3)

    source_gauss = np.empty([3, 2])
    beta = np.empty([3])
    
    d_s = np.empty([3])
    u_q = np.empty([3])

    alpha_w, omega_w = np.empty([3]), np.empty([3])

    if i_order == 0:
        for i in range(1, total_volumes - 1):  # For each control volume boundary
            slope = (u[i + 1, 0] - u[i, 0]) / h
            u_r_output[i] = u[i, 0] + slope * (z[i + 1, 0] - x[i, 0])
            d_u_r_output[i] = slope
            source_gauss[i, 0], source_gauss[i, 1] = u[i, 0], u[i, 0]
    elif i_order == 1:  # Linear reconstruction
        for i in range(1, total_volumes - 1):  # For each control volume
            slope1 = (u[i, 0] - u[i - 1, 0]) / h  # 1 st degree polynomial.Stencil[i - 1, i]
            slope = slope1
            slope2 = (u[i + 1, 0] - u[i, 0]) / h  # 1 st degree polynomial.Stencil[i, i + 1]
            if (abs(slope2) < abs(slope)):  # search of minimum slope (slope)
                slope = slope2
            u_l_output[i] = u[i, 0] + slope * (z[i, 0] - x[i, 0])
            u_r_output[i] = u[i, 0] + slope * (z[i + 1, 0] - x[i, 0])
            d_u_l_output[i], d_u_r_output[i] = slope, slope
            source_gauss[i, 0],source_gauss[i, 1] = u[i, 0], u[i, 0]
    elif i_order == 2:  # WENO - 5
        d_r[0], d_r[1], d_r[2] = 0.3, 0.6, 0.1
        d_l[0], d_l[1], d_l[2] = 0.1, 0.6, 0.3
        eps = 1E-06
        c1 = 13 / 12
        c2 = 1 / 4
        for i in range(2, total_volumes - 2):
            u_i, u_i_m1, u_i_m2 = u[i, 0], u[i - 1, 0], u[i - 2, 0]

            u_i1, u_i2 = u[i + 1, 0], u[i + 2, 0]

            u_r[2] = -7 / 6 * u_i_m1 + 11 / 6 * u_i + u_i_m2 / 3  # {i - 2, i - 1, i}
            u_l[2] = 5 / 6 * u_i_m1 + 1 / 3 * u_i - u_i_m2 / 6

            d_u_r[2] = (-3 * u_i_m1 + 2 * u_i + u_i_m2) / h  # {i - 2, i - 1, i}
            d_u_l[2] = (u_i - u_i_m1) / h

            u_l[1] = u_i_m1 / 3 + 5 / 6 * u_i - u_i1 / 6  # {i - 1, i, i + 1}
            u_r[1] = -u_i_m1 / 6 + 5 / 6 * u_i + u_i1 / 3

            d_u_l[1] = (u_i - u_i_m1) / h  # {i - 1, i, i + 1}
            d_u_r[1] = (u_i1 - u_i) / h

            u_r[0] = 5 / 6 * u_i1 + u_i / 3 - u_i2 / 6  # {i, i + 1, i + 2}
            u_l[0] = -7 / 6 * u_i1 + 11 * u_i / 6 + u_i2 / 3

            d_u_r[0] = (u_i1 - u_i) / h  # {i, i + 1, i + 2}
            d_u_l[0] = (-u_i2 - 2 * u_i + 3 * u_i1) / h

            # Piecewise parabolic reconstruction (r=3) - Smoothness indicators - slide 33
            beta[0] = c1 * (u_i - 2 * u_i1 + u_i2) ** 2 + c2 * (3 * u_i - 4 * u_i1 + u_i2) ** 2
            beta[1] = c1 * (u_i_m1 - 2 * u_i + u_i1) ** 2 + c2 * (u_i_m1 - u_i1) ** 2
            beta[2] = c1 * (u_i_m2 - 2 * u_i_m1 + u_i) ** 2 + c2 * (u_i_m2 - 4 * u_i_m1 + 3 * u_i) ** 2

            # Right extrapolated values
            u_r_output[i], d_u_r_output[i] = u_d_u_func(alpha_r, omega_r, d_r, u_r, d_u_r, eps, beta)
            # Left extrapolated values
            u_l_output[i], d_u_l_output[i] = u_d_u_func(alpha_l, omega_l, d_l, u_l, d_u_l, eps, beta)

            # WENO FOR REACTIVE TERMS
            # ** ** ** ** ** ** ** ** ** ** First Gaussian point ** ** ** ** ** ** ** ** ** **
            d_s[0] = (210 - math.sqrt(3)) / (1080)
            d_s[1] = 11 / 18
            d_s[2] = (210 + math.sqrt(3)) / (1080)

            alpha_w, omega_w = alpha_omega_func(alpha_w, omega_w, d_s, eps, beta)

            u_q[2] = u_i - (-4 * u_i_m1 + 3 * u_i + u_i_m2) * math.sqrt(3) / 12  # {i - 2, i - 1, i}
            u_q[1] = u_i - (-u_i_m1 + u_i1) * math.sqrt(3) / 12  # {i - 1, i, i + 1}
            u_q[0] = u_i + (u_i2 + 3 * u_i - 4 * u_i1) * math.sqrt(3) / 12  # {i, i + 1, i + 2}

            general_sum = 0
            for j in range(0, 2):
                general_sum += omega_w[j] * u_q[j]
            source_gauss[i, 1] = general_sum

            # ** ** ** ** ** ** ** ** ** ** Second Gaussian point ** ** ** ** ** ** ** ** ** **
            d_s[0] = (210 + math.sqrt(3)) / (1080)
            d_s[1] = 11 / 18
            d_s[2] = (210 - math.sqrt(3)) / (1080)

            alpha_w, omega_w = alpha_omega_func(alpha_w, omega_w, d_s, eps, beta)

            u_q[2] = u_i + (-4 * u_i_m1 + 3 * u_i + u_i_m2) * math.sqrt(3) / 12  #{i - 2, i - 1, i}
            u_q[1] = u_i + (-u_i_m1 + u_i1) * math.sqrt(3) / 12  #{i - 1, i, i + 1}
            u_q[0] = u_i - (u_i2 + 3 * u_i - 4 * u_i1) * math.sqrt(3) / 12  #{i, i + 1, i + 2}

            general_sum = 0
            for j in range(0, 2):
                general_sum += omega_w[j] * u_q[j]
            source_gauss[i, 1] = general_sum

    return u_l, u_r, d_u_l, d_u_r, source_gauss


def oprecons(i_order, total_volumes, x, z, h, dt, u, k_z, v_z, q_z, u_l, u_r, d_u_l, d_u_r, source_gauss):
    f_c_r = np.zeros(total_volumes)
    f_c_l = np.zeros(total_volumes)
    f_d_r = np.zeros(total_volumes)
    f_d_l = np.zeros(total_volumes)
    f_c_lf = np.zeros(total_volumes)
    f_c_lw = np.zeros(total_volumes)
    f_c_fo = np.zeros(total_volumes)
    f_d_m = np.zeros(total_volumes)
    l = np.zeros(total_volumes - 3)

    f = np.empty([3])

    if i_order == 0:
        for i in range(2, total_volumes - 2):
            f[i] = k_z[i] * d_u_r[i] - v_z[i] * u_r[i]
    else:
        for i in range(2, total_volumes - 2):
            f_c_l[i] = v_z[i] * u_l[i]
            f_c_r[i] = v_z[i + 1] * u_r[i]
            f_d_l[i] = k_z[i] * d_u_l[i]
            f_d_r[i] = k_z[i + 1] * d_u_r[i]

   # CONVECTIVE FLUX FORCE
    for i in range(2, total_volumes - 2):
        f_c_lf[i] = (f_c_r[i] + f_c_l[i + 1]) / 2 + h / (2 * dt) * (u_r[i] - u_l[i + 1])
    for i in range(2, total_volumes - 2):
        u_l_w = (u_l[i + 1, 1] + u_r[i, 1]) / 2 + dt / (2 * h) * (f_c_r[i] - f_c_l[i + 1])
        f_c_lw[i] = v_z[i + 1] * u_l_w
    for i in range(2, total_volumes - 2):
        f_c_fo[i] = (f_c_lw[i] + f_c_lf[i]) / 2

    # DIFFUSIVE FLUX Arithmetic mean
    for i in range(2, total_volumes - 2):
        f_d_m[i] = (f_d_l[i + 1] + f_d_r[i]) / 2

    for i in range(2, total_volumes - 2):
        f[i] = f_d_m[i] - f_c_fo[i]

    f[0] = -f_c_l[0] + f_d_l[0]
    f[total_volumes + 1] = -f_c_l[total_volumes + 1] + f_d_l[total_volumes + 1]

    # REACTION Operator:
    for i in range(2, total_volumes - 2):
        q_u = q_z[i] * 1 / 2 * (source_gauss[i, 0] + source_gauss[i, 1])
        l[i] = (f[i] - f[i - 1]) / h - q_u

    return l


def fv_rk3(i_order, n_volumes, a, b, final_t, co, alpha,
           discontinuity_point, value_right, value_left, dimensions=1):

    total_volumes = n_volumes + 6
    """Function to calculate a third order Runge-Kutta scheme as part of a finite volume scheme

    Parameters:
    n_v (int): number of control volumes
    n_t: number of time steps
    l: length of the domain
    final_t: output time
    u_init_cond: vector with the initial conditions
    u_0: solution in the left boundary (Dirichlet)
    U_1: solution in the right boundary (Dirichlet)
    v: velocity function
    k: diffusion coefficient function
    u_sol: vector with the solution

    Auxiliary:
    a1: limit for y-axis
    a2: limit for y-axis


    Returns:
    x: vector with nodal coordinates
    u: vector containing the solution
    u_fig: solution

   """

    # ############################################ Defining variables ######################################################
    # Will leave the 1 here even if it's not necessary in 1D but potentially when I move to more dimensions
    x_n = np.zeros((n_volumes + 6, dimensions)).reshape(n_volumes + 6, dimensions)

    z = np.zeros((n_volumes + 7, dimensions)).reshape(n_volumes + 7, dimensions)
    u = np.zeros((n_volumes + 6, dimensions)).reshape(n_volumes + 6, dimensions)
    u_a = np.zeros((n_volumes + 6, dimensions)).reshape(n_volumes + 6, dimensions)
    u_fig = np.zeros(n_volumes)
    x_fig = np.zeros(n_volumes)

    uaux = np.empty([n_volumes, 1])

    # Vectors for physical data
    k_z, v_z, q_z = np.zeros_like(z), np.zeros_like(z), np.zeros_like(z)

    h = (b - a) / n_volumes

    # ############################################ x_n initialization ##################################################
    x_n_start = (a - h / 2) - 2 * h
    x_n_stop = -x_n_start + b
    x_n = np.linspace(start=x_n_start, stop=x_n_stop, num=len(x_n)).reshape(n_volumes + 6, dimensions)
    x_fig = x_n[3:-3]

    # ############################################ z initialization ####################################################
    z_start = (a - h) - 2 * h
    z_stop = b + 3 * h
    z = np.linspace(start=z_start, stop=z_stop, num=len(z)).reshape(len(z), dimensions)

    # ############################################ Average of the initial conditions ###################################
    for i in range(3, n_volumes + 3):
        z1 = z[i, 0]
        z2 = z[i + 1, 0]

        solution = gauss_sol(z1=z1, z2=z2, h=h, n=8, discontinuity_point=discontinuity_point,
                             value_right=value_right, value_left=value_left)
        u_a[i, 0] = solution

    u_fig = u_a[3:-3]

    # ############################################ Physical data #######################################################
    for i in range(0, n_volumes + 7):
        k_z[i] = physical_data_k(z[i, 0])  # Diffusion coefficient
        v_z[i] = physical_data_v(z[i, 0])  # Velocity
        q_z[i] = physical_data_q(z[i, 0])  # Reaction coefficient

    # Time step
    delta_t = 1000

    for i in range(0, n_volumes):

        if abs(v_z[i]) > 0:
            deltat_c = co * h / v_z[i]
            if deltat_c < delta_t:
                delta_t = deltat_c
        else:
            delta_t = 1000000

        if k_z[i] > 0:
            delta_t_d = alpha * h * h / k_z[i]
            if delta_t_d < delta_t:
                delta_t = delta_t_d

    n_t = math.floor(final_t / delta_t)

    # Coefficients for Runge - Kutta TVD
    c1 = 1 / 3
    c2 = delta_t / 4
    c3 = 2 * delta_t / 3

    a1 = -0.1
    a2 = 2

    # ############################################ RK3-TVD Step 1 ######################################################
    # Neumann homogeneous boundary conditions
    # This is weird, is there another way to assign this???
    u_a[2, 0] = u_a[3, 0]
    u_a[1, 0] = u_a[4, 0]
    u_a[0, 0] = u_a[5, 0]
    # This is weird, is there another way to assign this??? NOT SURE ABOUT WHAT IS HAPPENING HERE
    u_a[n_volumes + 3, 0] = u_a[n_volumes + 2, 1]
    u_a[n_volumes + 4, 0] = u_a[n_volumes + 1, 1]
    u_a[n_volumes + 5, 0] = u_a[n_volumes + 0, 1]

    u_l, u_r, du_l, du_r, source_gauss = recons(i_order, n_volumes, u_a, h, x_n, z)

    L = oprecons(i_order, n_volumes, x_n, z, h, delta_t, u_a, k_z, v_z, q_z, u_l, u_r, du_l, du_r, source_gauss)

    for i in range(2, n_volumes - 2):
        uaux[i, 0] = u_a[i, 0] + delta_t * L[i]

    # ############################################ RK3-TVD Step 2 ######################################################


    # ############################################ RK3-TVD Step 3 ######################################################



    return x_fig, u_fig