import math

import numpy as np


def uinic(x, discontinuity_point_, value_left_, value_right_):
    if (x >= 0) and (x <= discontinuity_point_):
        value_ = value_left_
    else:
        value_ = value_right_

    return value_


def gauss_legendre(n=96, a_=-1.0, b_=1.0):
    """Function to obtain the Gauss-Legendre quadrature roots and weights
    :param n: nth polynomial
    :param a_: range starting point [a_, b_]
    :param b_: range end point [a_, b_]
    :return:
    x_: roots of the nth legendre polynomial, transformed to range [a_, b_]
    w_: quadrature weights
    """

    assert(n >= 1)  # Number of sample points and weights must be >= 1
    # Get the roots and quadrature weights
    x_, w_ = np.polynomial.legendre.leggauss(deg=n)

    # Transform to [a, b] range
    x_ = (a_ + b_) / 2 + (b_ - a_) / 2 * x_
    w_ = (b_ - a_) / 2 * w_

    return x_, w_


def gauss_sol(z1_, z2_, h_, discontinuity_point_, value_right_, value_left_, n=8):
    x_, w_ = gauss_legendre(n, z1_, z2_)

    sol = 0
    for i_ in range(0, n):
        sol += w_[i_] * uinic(x_[i_], discontinuity_point_, value_right_, value_left_)
        sol = sol / h_

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


def fv_rk3(i_order_, number_of_volumes, a_, b_, final_t, co, alpha,
           discontinuity_point, value_right, value_left, dimensions=1):

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
    x_n = np.zeros((number_of_volumes + 6, dimensions)).reshape(number_of_volumes + 6, dimensions)

    z = np.zeros((number_of_volumes + 7, dimensions)).reshape(number_of_volumes + 7, dimensions)
    u = np.zeros((number_of_volumes + 6, dimensions)).reshape(number_of_volumes + 6, dimensions)
    u_a = np.zeros((number_of_volumes + 6, dimensions)).reshape(number_of_volumes + 6, dimensions)
    u_fig = np.zeros_like(u_a)
    x_fig = np.zeros_like(u_fig)

    # Vectors for physical data
    k_z, v_z, q_z = np.zeros_like(z), np.zeros_like(z), np.zeros_like(z)

    h_ = (b_ - a_) / number_of_volumes

    # ############################################ x_n initialization ##################################################
    x_n_start = (a_ - h_ / 2) - 2 * h_
    x_n_stop = -x_n_start + b_
    x_n = np.linspace(start=x_n_start, stop=x_n_stop, num=len(x_n)).reshape(number_of_volumes + 6, dimensions)

    # ############################################ z initialization ####################################################
    z_start = (a_ - h_) - 2 * h_
    z_stop = b_ + 3 * h_
    z = np.linspace(start=z_start, stop=z_stop, num=len(z)).reshape(len(z), dimensions)

    # ############################################ Average of the initial conditions ###################################
    for i in range(3, number_of_volumes + 3):
        z1 = z[i, 0]
        z2 = z[i + 1, 0]
        solution = gauss_sol(z1, z2, h_, discontinuity_point, value_right, value_left, n=8)
        u_a[i, 0] = solution

    # ############################################ Physical data #######################################################
    for i in range(0, number_of_volumes + 7):
        k_z[i] = physical_data_k(z[i, 0])  # Diffusion coefficient
        v_z[i] = physical_data_v(z[i, 0])  # Velocity
        q_z[i] = physical_data_q(z[i, 0])  # Reaction coefficient

    # Time step
    delta_t = 1000

    for i in range(0, number_of_volumes):

        if (abs(v_z[i]) > 0):
            delta_t_c = co * h_ / v_z[i]
            if (delta_t_c < delta_t):
                delta_t = delta_t_c
        else:
            delta_t = 1000000

        if (k_z[i] > 0):
            delta_t_d = alpha * h_ * h_ / k_z[i]
            if (delta_t_d < delta_t):
                delta_t = delta_t_d

    n_t = math.floor(final_t / delta_t)

    # Coefficients for Runge - Kutta TVD
    c1 = 1 / 3
    c2 = delta_t / 4
    c3 = 2 * delta_t / 3

    a1 = -0.1
    a2 = 2

    j = 0
    for i in range(3, number_of_volumes + 3):
        u_fig[j, 0] = u_a[i, 0]
        j += 1

    j = 1
    for i in range(3, number_of_volumes + 3):
        x_fig[j, 0] = x_n[i, 0]
        j = j + 1

    return a1, a2, u_fig, x_fig
